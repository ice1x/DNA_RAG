"""Core DNA analysis engine.

This module contains :class:`DNAAnalysisEngine` -- the heart of **dna-rag**.
It orchestrates a two-step LLM pipeline:

1. **SNP identification** -- ask the LLM which SNPs are relevant to the
   user's question.
2. **Interpretation** -- filter the user's DNA data against those SNPs,
   then ask the LLM to explain the results.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import DataFrame

from dna_rag.cache.base import Cache
from dna_rag.exceptions import (
    LLMResponseError,
    NoMatchingVariantsError,
    NoSNPsFoundError,
)
from dna_rag.llm.base import LLMProvider
from dna_rag.logging import get_logger
from dna_rag.models import AnalysisResult, SNPDictResponse, SNPMetadata, SNPResult
from dna_rag.parsers.detector import detect_and_parse

if TYPE_CHECKING:
    from dna_rag.snp_database import SNPDatabase
    from dna_rag.vector_store import SNPVectorStore

logger = get_logger(__name__)

# Cache namespace constants
NS_FILE = "dna_file"
NS_SNP = "snp_dict"
NS_ANSWER = "answer"


# Regex to match ```json ... ``` or ``` ... ```
_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", flags=re.DOTALL,
)


class DNAAnalysisEngine:
    """RAG-augmented LLM pipeline for personal DNA analysis.

    The engine is **LLM-agnostic**: each pipeline step can use a different
    LLM provider (or the same one).  An optional vector store provides
    RAG context for SNP identification, and an optional SNP database
    validates RSIDs against NCBI dbSNP.

    Args:
        snp_llm: LLM used for **Step 1** -- SNP identification.
        interpretation_llm: LLM used for **Step 2** -- result interpretation.
            If ``None``, *snp_llm* is used for both steps.
        cache: A cache backend implementing
            :class:`~dna_rag.cache.base.Cache`.  Pass ``None`` to disable.
        vector_store: Optional :class:`~dna_rag.vector_store.SNPVectorStore`
            for RAG-augmented SNP identification.
        snp_database: Optional :class:`~dna_rag.snp_database.SNPDatabase`
            for NCBI validation of LLM-identified RSIDs.
        rag_search_results: Max number of vector store results to use as context.
        rag_min_similarity: Minimum similarity threshold for vector store results.
    """

    def __init__(
        self,
        snp_llm: LLMProvider,
        interpretation_llm: LLMProvider | None = None,
        cache: Cache | None = None,
        vector_store: SNPVectorStore | None = None,
        snp_database: SNPDatabase | None = None,
        rag_search_results: int = 10,
        rag_min_similarity: float = 0.3,
        medical_disclaimer: str = "",
    ) -> None:
        self._snp_llm = snp_llm
        self._interp_llm = interpretation_llm or snp_llm
        self._cache = cache
        self._vector_store = vector_store
        self._snp_db = snp_database
        self._rag_search_results = rag_search_results
        self._rag_min_similarity = rag_min_similarity
        self._medical_disclaimer = medical_disclaimer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, question: str, dna_file: Path) -> AnalysisResult:
        """Run the full analysis pipeline.

        Args:
            question: The user's question, e.g. ``"lactose tolerance"``.
            dna_file: Path to a raw DNA data file (23andMe / AncestryDNA /
                MyHeritage).

        Returns:
            :class:`~dna_rag.models.AnalysisResult` containing matched
            SNPs and the LLM interpretation.

        Raises:
            NoSNPsFoundError: The LLM could not identify relevant SNPs.
            NoMatchingVariantsError: None of the identified SNPs appear
                in the user's DNA file.
            LLMResponseError: The LLM returned an unparseable response.
            FileNotFoundError: *dna_file* does not exist.
            UnsupportedFormatError: DNA file format not recognised.
        """
        logger.info("analysis_start", question=question, file=str(dna_file))

        file_hash = self._hash_file(dna_file)
        cache_key = f"{file_hash}:{question}"

        # 1) Check full-answer cache
        if self._cache is not None:
            cached: AnalysisResult | None = self._cache.get(NS_ANSWER, cache_key)
            if cached is not None:
                logger.info("cache_hit", namespace=NS_ANSWER)
                # Return a copy with cached=True to avoid mutating the stored object
                return cached.model_copy(update={"cached": True})

        # 2) Load & parse DNA file (cached by hash)
        df = self._load_dna(dna_file, file_hash)

        # 3) Get RAG context from vector store (optional)
        rag_context = self._get_rag_context(question)

        # 4) Ask LLM for relevant SNPs (with optional RAG context)
        snp_response = self._get_snp_dict(question, rag_context=rag_context)
        if not snp_response.snps:
            raise NoSNPsFoundError(
                f"No relevant SNPs identified for question: {question}"
            )
        llm_snp_count = len(snp_response.snps)

        # 5) Validate SNPs against NCBI (optional)
        validated_snps = self._validate_snps(snp_response.snps)
        if not validated_snps:
            raise NoSNPsFoundError(
                f"No valid SNPs identified for question: {question}"
            )
        snp_response = SNPDictResponse(snps=validated_snps)

        # 6) Filter DNA data
        matched_df = self._filter_snps(df, snp_response.snps)
        if matched_df.empty:
            raise NoMatchingVariantsError(
                f"No matching variants found in DNA file for: {question}"
            )

        # 7) Interpret
        matched_snps = self._dataframe_to_snp_results(matched_df)
        interpretation = self._interpret(matched_df, question)

        result = AnalysisResult(
            question=question,
            matched_snps=matched_snps,
            interpretation=interpretation,
            snp_count_requested=llm_snp_count,
            snp_count_matched=len(matched_snps),
            cached=False,
            rag_context_used=bool(rag_context),
            validation_used=self._snp_db is not None,
        )

        if self._cache is not None:
            self._cache.set(NS_ANSWER, cache_key, result)

        logger.info(
            "analysis_complete",
            snps_requested=result.snp_count_requested,
            snps_matched=result.snp_count_matched,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(dna_file: Path) -> str:
        """Compute SHA-256 hash of *dna_file* for cache invalidation."""
        return hashlib.sha256(dna_file.read_bytes()).hexdigest()

    def _load_dna(self, dna_file: Path, file_hash: str) -> DataFrame:
        """Load and parse a DNA file, using cache if available."""
        if self._cache is not None:
            cached_df: DataFrame | None = self._cache.get(NS_FILE, file_hash)
            if cached_df is not None:
                logger.info("cache_hit", namespace=NS_FILE)
                return cached_df

        df = detect_and_parse(dna_file)
        logger.info("dna_loaded", rows=len(df), file=str(dna_file))

        if self._cache is not None:
            self._cache.set(NS_FILE, file_hash, df)
        return df

    def _get_rag_context(self, question: str) -> str:
        """Query the vector store for SNPs semantically related to *question*.

        Returns a formatted string suitable for injection into the LLM
        prompt, or an empty string if no vector store is configured or
        no results are found.
        """
        if self._vector_store is None:
            return ""

        try:
            results: dict[str, Any] = self._vector_store.search(
                question,
                n_results=self._rag_search_results,
                min_similarity=self._rag_min_similarity,
            )
        except Exception:
            logger.warning("vector_store_search_failed", question=question, exc_info=True)
            return ""

        if not results:
            return ""

        lines = ["Known SNPs that may be relevant (from database):"]
        for rsid, meta in results.items():
            gene = meta.get("gene", "unknown")
            trait = meta.get("trait", "unknown")
            sim = meta.get("similarity", 0)
            lines.append(f"  - {rsid}: gene={gene}, trait={trait} (similarity={sim:.2f})")

        logger.info("rag_context_found", count=len(results))
        return "\n".join(lines)

    def _validate_snps(
        self, snps: dict[str, SNPMetadata],
    ) -> dict[str, SNPMetadata]:
        """Validate RSIDs against NCBI dbSNP, removing invalid ones.

        When a valid SNP is found, the LLM-provided gene name is replaced
        with the authoritative gene from dbSNP.  Validation metadata
        (MAF, ClinVar significance) is stored in ``_validation_results``
        for later use in the interpretation prompt.

        If no SNP database is configured or validation fails, returns
        *snps* unchanged (graceful degradation).
        """
        if self._snp_db is None:
            return snps

        try:
            results = self._snp_db.validate_batch(list(snps.keys()))
        except Exception:
            logger.warning("snp_validation_failed", exc_info=True)
            return snps

        valid: dict[str, SNPMetadata] = {}
        invalid: list[str] = []

        for rsid, meta in snps.items():
            validation = results.get(rsid)
            if validation is not None and validation.exists:
                # Overwrite LLM gene with authoritative dbSNP gene
                if validation.gene:
                    if meta.gene.lower() != validation.gene.lower():
                        logger.info(
                            "gene_corrected",
                            rsid=rsid,
                            llm_gene=meta.gene,
                            dbsnp_gene=validation.gene,
                        )
                    meta = meta.model_copy(update={"gene": validation.gene})
                valid[rsid] = meta
            else:
                invalid.append(rsid)

        if invalid:
            logger.info(
                "snps_filtered_by_validation",
                invalid=invalid,
                kept=len(valid),
                removed=len(invalid),
            )

        # Store validation results for use in interpretation prompt
        self._validation_results = results
        return valid

    def _get_snp_dict(
        self, question: str, *, rag_context: str = "",
    ) -> SNPDictResponse:
        """Query the LLM for SNP identifiers related to *question*.

        When *rag_context* is non-empty it is injected into the prompt as
        additional hints from the vector store.
        """
        if self._cache is not None:
            cached_snp: SNPDictResponse | None = self._cache.get(NS_SNP, question)
            if cached_snp is not None:
                logger.info("cache_hit", namespace=NS_SNP)
                return cached_snp

        prompt = (
            "You are an expert in human genetics and genomics. "
            "Given the user's question, identify the most relevant SNPs "
            "(Single Nucleotide Polymorphisms) and return them as a JSON dictionary.\n\n"
            "IMPORTANT RULES:\n"
            "- Always try your best to find relevant SNPs, even for complex "
            "polygenic traits such as ancestry, ethnicity, population genetics, "
            "disease risk, or behavioural traits.\n"
            "- For ancestry and ethnicity questions, include well-known "
            "ancestry-informative markers (AIMs), population-specific alleles, "
            "and haplogroup-associated SNPs.\n"
            "- Never refuse to answer based on topic sensitivity. "
            "This is a legitimate personal genomics tool and the user is "
            "analysing their own DNA data.\n"
            "- Return at least 3-5 SNPs whenever possible.\n\n"
            f"Question: {question}\n\n"
        )

        if rag_context:
            prompt += (
                "The following SNPs from our database may be relevant. "
                "Consider these as suggestions, but also use your own knowledge "
                "to determine the most appropriate SNPs. You may include "
                "these or choose different ones:\n\n"
                f"{rag_context}\n\n"
            )

        prompt += (
            "Return ONLY valid JSON in this exact format "
            "(no markdown, no explanation):\n"
            "{\n"
            '  "rs12345": {\n'
            '    "gene": "GENE_NAME",\n'
            '    "chromosome": "1",\n'
            '    "position": 12345,\n'
            '    "trait": "Associated trait description"\n'
            "  }\n"
            "}\n"
        )

        raw = self._snp_llm.invoke(prompt)
        json_str = self.extract_json(raw)

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "llm_json_parse_error",
                raw_response=raw[:200],
                error=str(exc),
            )
            raise LLMResponseError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseError(
                f"Expected JSON object from LLM, got {type(parsed).__name__}"
            )

        # Validate through Pydantic -- coerce values into SNPMetadata
        snp_map: dict[str, SNPMetadata] = {}
        for k, v in parsed.items():
            if isinstance(v, dict):
                try:
                    snp_map[k] = SNPMetadata(**v)
                except Exception:
                    logger.warning("skipping_invalid_snp", rsid=k, raw=v)
            # non-dict values silently skipped

        snp_response = SNPDictResponse(snps=snp_map)

        if self._cache is not None:
            self._cache.set(NS_SNP, question, snp_response)
        return snp_response

    @staticmethod
    def extract_json(text: str) -> str:
        """Extract JSON from an LLM response.

        Handles responses wrapped in markdown code blocks
        (````json ... ````).  If no code block is found the whole
        text is returned stripped.
        """
        text = text.strip()
        match = _CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _filter_snps(
        df: DataFrame,
        snps: dict[str, SNPMetadata],
    ) -> DataFrame:
        """Filter *df* to rows whose RSID appears in *snps*.

        Merges the LLM-provided metadata (gene, trait) onto the
        matched rows.
        """
        if not snps:
            return DataFrame()

        rsid_set = set(snps.keys())
        matched = df[df["RSID"].str.lower().isin(rsid_set)].copy()
        if matched.empty:
            return matched

        meta_records = [
            {"rsid_key": rsid, "gene": m.gene, "trait": m.trait}
            for rsid, m in snps.items()
        ]
        meta_df = pd.DataFrame(meta_records)
        matched["rsid_key"] = matched["RSID"].str.lower()
        result = matched.merge(meta_df, on="rsid_key", how="left")
        result = result.drop(columns=["rsid_key"])
        return result

    @staticmethod
    def _dataframe_to_snp_results(df: DataFrame) -> list[SNPResult]:
        """Convert matched DataFrame rows to :class:`SNPResult` models."""
        results: list[SNPResult] = []
        for _, row in df.iterrows():
            results.append(
                SNPResult(
                    rsid=str(row["RSID"]),
                    chromosome=str(row["CHROMOSOME"]),
                    position=int(row["POSITION"]),
                    genotype=str(row["GENOTYPE"]),
                    gene=str(row.get("gene", "unknown")),
                    trait=str(row.get("trait", "unknown")),
                )
            )
        return results

    def _interpret(self, df: DataFrame, question: str) -> str:
        """Ask the LLM to interpret the matched SNP data."""
        data_str = df.to_string(index=False)

        # Build validated metadata block if available
        validated_info = self._build_validated_info(df)

        prompt = (
            "You are a genetics expert providing personalised DNA analysis. "
            "The following SNP data was extracted from a user's DNA file "
            f"based on their question: {question}\n\n"
            f"{data_str}\n\n"
        )

        if validated_info:
            prompt += (
                "VERIFIED DATA from NCBI databases (use this as ground truth, "
                "do NOT contradict it):\n"
                f"{validated_info}\n\n"
            )

        prompt += (
            "Provide a clear, concise interpretation for the user. "
            "Include:\n"
            "1. What each relevant genotype means\n"
            "2. The overall assessment for the trait in question\n"
            "Keep the response under 500 words.\n\n"
            "CRITICAL RULES:\n"
            "- ONLY use gene names from the VERIFIED DATA above. "
            "Do NOT invent or guess gene associations.\n"
            "- If a SNP has no known clinical significance in ClinVar, "
            "explicitly state that.\n"
            "- If a SNP has a minor allele frequency (MAF) above 5%, "
            "note that it is a common variant.\n"
            "- If the evidence linking a SNP to the asked trait is weak "
            "or non-existent, say so honestly instead of speculating.\n"
            "- Do NOT claim a SNP is 'associated with' a trait unless "
            "there is well-established research supporting it.\n\n"
            "IMPORTANT: You MUST respond in the SAME language as the user's "
            "question above. If the question is in Russian, respond in Russian. "
            "If in English, respond in English. Match the language exactly."
        )
        if self._medical_disclaimer:
            prompt += (
                "\n\nAt the end of your response, include the following disclaimer "
                "translated into the same language as your response:\n"
                f"{self._medical_disclaimer}"
            )
        return self._interp_llm.invoke(prompt)

    def _build_validated_info(self, df: DataFrame) -> str:
        """Build a block of verified SNP metadata from NCBI validation results.

        Includes authoritative gene name, MAF, and ClinVar significance
        for each matched SNP.
        """
        results = getattr(self, "_validation_results", None)
        if not results:
            return ""

        lines: list[str] = []
        for _, row in df.iterrows():
            rsid = str(row["RSID"]).lower()
            vr = results.get(rsid)
            if vr is None or not vr.exists:
                continue
            parts = [f"- {rsid}:"]
            if vr.gene:
                parts.append(f"gene={vr.gene}")
            if vr.maf is not None:
                parts.append(f"MAF={vr.maf:.4f} ({vr.maf_allele}, {vr.maf_study})")
            if vr.clinical_significance:
                parts.append(f"ClinVar={vr.clinical_significance}")
                if vr.clinvar_trait:
                    parts.append(f"trait={vr.clinvar_trait}")
            else:
                parts.append("ClinVar=not found")
            lines.append(" ".join(parts))

        return "\n".join(lines)
