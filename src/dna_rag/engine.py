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
    """Two-step LLM pipeline for personal DNA analysis.

    The engine is **LLM-agnostic**: each pipeline step can use a different
    LLM provider (or the same one).

    Args:
        snp_llm: LLM used for **Step 1** -- SNP identification.
        interpretation_llm: LLM used for **Step 2** -- result interpretation.
            If ``None``, *snp_llm* is used for both steps.
        cache: A cache backend implementing
            :class:`~dna_rag.cache.base.Cache`.  Pass ``None`` to disable.

    Example::

        engine = DNAAnalysisEngine(
            snp_llm=DeepSeekProvider(settings),           # reasoning model
            interpretation_llm=OpenAICompatProvider(s2),   # cheaper model
            cache=InMemoryCache(),
        )
    """

    def __init__(
        self,
        snp_llm: LLMProvider,
        interpretation_llm: LLMProvider | None = None,
        cache: Cache | None = None,
    ) -> None:
        self._snp_llm = snp_llm
        self._interp_llm = interpretation_llm or snp_llm
        self._cache = cache

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

        # 3) Ask LLM for relevant SNPs
        snp_response = self._get_snp_dict(question)
        if not snp_response.snps:
            raise NoSNPsFoundError(
                f"No relevant SNPs identified for question: {question}"
            )

        # 4) Filter DNA data
        matched_df = self._filter_snps(df, snp_response.snps)
        if matched_df.empty:
            raise NoMatchingVariantsError(
                f"No matching variants found in DNA file for: {question}"
            )

        # 5) Interpret
        matched_snps = self._dataframe_to_snp_results(matched_df)
        interpretation = self._interpret(matched_df, question)

        result = AnalysisResult(
            question=question,
            matched_snps=matched_snps,
            interpretation=interpretation,
            snp_count_requested=len(snp_response.snps),
            snp_count_matched=len(matched_snps),
            cached=False,
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

    def _get_snp_dict(self, question: str) -> SNPDictResponse:
        """Query the LLM for SNP identifiers related to *question*."""
        if self._cache is not None:
            cached_snp: SNPDictResponse | None = self._cache.get(NS_SNP, question)
            if cached_snp is not None:
                logger.info("cache_hit", namespace=NS_SNP)
                return cached_snp

        prompt = (
            "You are an expert in human genetics. Given the user's question, "
            "identify the most relevant SNPs (Single Nucleotide Polymorphisms) "
            "and return them as a JSON dictionary.\n\n"
            f"Question: {question}\n\n"
            "Return ONLY valid JSON in this exact format "
            "(no markdown, no explanation):\n"
            "{\n"
            '  "rs12345": {\n'
            '    "gene": "GENE_NAME",\n'
            '    "chromosome": "1",\n'
            '    "position": 12345,\n'
            '    "trait": "Associated trait description"\n'
            "  }\n"
            "}\n\n"
            "If no relevant SNPs exist, return: {}\n"
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
        prompt = (
            "You are a genetics expert providing personalised DNA analysis. "
            "The following SNP data was extracted from a user's DNA file "
            f"based on their question: {question}\n\n"
            f"{data_str}\n\n"
            "Provide a clear, concise interpretation for the user. "
            "Include:\n"
            "1. What each relevant genotype means\n"
            "2. The overall assessment for the trait in question\n"
            "3. Any important caveats about genetic interpretation\n"
            "Keep the response under 500 words."
        )
        return self._interp_llm.invoke(prompt)
