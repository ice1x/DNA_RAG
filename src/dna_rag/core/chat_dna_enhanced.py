"""Enhanced ChatDNA with RAG, validation, and structured output.

This module extends the original ChatDNA with:
- Vector store for RAG-based SNP retrieval
- SNP validation through dbSNP
- Conversation history
- Structured output with confidence scores
- Source citations
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from langchain.schema import HumanMessage
from pandas import DataFrame
from pydantic import BaseModel, Field

from dna_rag.utils.langchain_deepseek import ChatDeepSeek
from dna_rag.utils.snp_database import SNPDatabase
from dna_rag.utils.vector_store import SNPVectorStore, create_sample_snp_database

logger = logging.getLogger(__name__)


class SNPResult(BaseModel):
    """Individual SNP result with metadata."""

    rsid: str
    genotype: str
    gene: str | None = None
    trait: str | None = None
    validated: bool = False
    similarity: float = 0.0


class InterpretationResult(BaseModel):
    """Structured interpretation result."""

    question: str
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    snps_found: list[SNPResult] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """Single conversation message."""

    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class ChatDNAEnhanced:
    """Enhanced ChatDNA with RAG and validation capabilities.

    This class extends the original ChatDNA with:
    - Vector-based SNP retrieval (RAG)
    - SNP validation through dbSNP
    - Conversation history
    - Structured outputs with confidence scores
    """

    def __init__(
        self,
        api_key: str,
        use_vector_store: bool = True,
        use_validation: bool = True,
        vector_store_path: Path | None = None,
    ) -> None:
        """Initialize enhanced ChatDNA.

        Parameters
        ----------
        api_key : str
            DeepSeek API key.
        use_vector_store : bool
            Whether to use vector store for RAG.
        use_validation : bool
            Whether to validate SNPs through dbSNP.
        vector_store_path : Optional[Path]
            Path to persist vector store.
        """
        self.llm: ChatDeepSeek = ChatDeepSeek(
            model="deepseek-r1:free",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )

        # Caches
        self._dna_cache: dict[Path, tuple[str, DataFrame]] = {}
        self._answer_cache: dict[str, dict[str, InterpretationResult]] = {}

        # Enhanced components
        self._use_vector_store = use_vector_store
        self._use_validation = use_validation

        # Initialize vector store
        self._vector_store: SNPVectorStore | None = None
        if use_vector_store:
            self._vector_store = SNPVectorStore(persist_directory=vector_store_path)
            # Initialize with sample data if empty
            if self._vector_store.count() == 0:
                create_sample_snp_database(self._vector_store)

        # Initialize SNP database
        self._snp_db: SNPDatabase | None = None
        if use_validation:
            self._snp_db = SNPDatabase()

        # Conversation history (per file)
        self._conversations: dict[str, list[ConversationMessage]] = {}

    def ask(
        self, question: str, dna_file: Path, return_structured: bool = True
    ) -> str | InterpretationResult:
        """Answer a question using DNA data with enhanced features.

        Parameters
        ----------
        question : str
            The user's question.
        dna_file : Path
            Path to DNA file.
        return_structured : bool
            Whether to return structured result.

        Returns
        -------
        Union[str, InterpretationResult]
            Answer as string or structured result.
        """
        file_hash = self._hash_file(dna_file)
        df = self._load_dna(dna_file, file_hash)

        # Add to conversation history
        conv_key = str(dna_file)
        if conv_key not in self._conversations:
            self._conversations[conv_key] = []
        self._conversations[conv_key].append(ConversationMessage(role="user", content=question))

        # Check cache
        answers_for_file = self._answer_cache.setdefault(file_hash, {})
        if question in answers_for_file:
            cached = answers_for_file[question]
            return cached if return_structured else cached.interpretation

        # Get relevant SNPs using vector store or LLM
        if self._use_vector_store and self._vector_store:
            snp_dict = self._get_snps_from_vector_store(question)
        else:
            snp_dict = self._get_snps_from_llm(question)

        if not snp_dict:
            result = InterpretationResult(
                question=question,
                interpretation="No relevant SNPs found for this question.",
                confidence=0.0,
                caveats=[
                    "Unable to find relevant genetic variants for this query.",
                    "This may be because the trait is not well-studied or the "
                    "question is too general.",
                ],
            )
            answers_for_file[question] = result
            self._conversations[conv_key].append(
                ConversationMessage(
                    role="assistant",
                    content=result.interpretation,
                    metadata={"confidence": result.confidence},
                )
            )
            return result if return_structured else result.interpretation

        # Validate SNPs if enabled
        if self._use_validation and self._snp_db:
            snp_dict = self._validate_snps(snp_dict)

        # Filter user's DNA
        filtered = self._filter_snps(df, snp_dict)

        if filtered.empty:
            result = InterpretationResult(
                question=question,
                interpretation="No matching variants found in your DNA file.",
                confidence=0.0,
                caveats=[
                    "Your DNA file may not contain the relevant SNPs.",
                    "This could be due to different genotyping arrays or file "
                    "format differences.",
                ],
            )
            answers_for_file[question] = result
            self._conversations[conv_key].append(
                ConversationMessage(
                    role="assistant",
                    content=result.interpretation,
                    metadata={"confidence": result.confidence},
                )
            )
            return result if return_structured else result.interpretation

        # Interpret with context
        result = self._interpret_enhanced(filtered, question, snp_dict)
        answers_for_file[question] = result

        self._conversations[conv_key].append(
            ConversationMessage(
                role="assistant",
                content=result.interpretation,
                metadata={"confidence": result.confidence},
            )
        )

        return result if return_structured else result.interpretation

    def get_conversation_history(self, dna_file: Path) -> list[ConversationMessage]:
        """Get conversation history for a DNA file.

        Parameters
        ----------
        dna_file : Path
            Path to DNA file.

        Returns
        -------
        List[ConversationMessage]
            Conversation history.
        """
        return self._conversations.get(str(dna_file), [])

    def _get_snps_from_vector_store(self, question: str) -> dict[str, dict[str, str | int | float]]:
        """Get relevant SNPs using vector store RAG.

        Parameters
        ----------
        question : str
            User's question.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            SNP dictionary.
        """
        if not self._vector_store:
            return {}

        return self._vector_store.search(question, n_results=10, min_similarity=0.3)

    def _get_snps_from_llm(self, question: str) -> dict[str, dict[str, str | int | float]]:
        """Get SNPs using LLM (legacy method).

        Parameters
        ----------
        question : str
            User's question.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            SNP dictionary.
        """
        prompt = f"""You are an expert in genetics. Based on the user's question,
create a dictionary describing relevant SNPs in JSON format.

Question: {question}

The dictionary should look like:
{{
    "rs123": {{
        "gene": "GENE", "chromosome": "1", "position": 1234,
        "trait": "Associated trait"
    }}
}}
Return an empty dictionary ({{}}) if nothing matches."""

        try:
            response = self.llm([HumanMessage(content=prompt)]).content.strip()
            data = json.loads(response)
            return data
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            logger.warning(f"Failed to get SNPs from LLM: {exc}")
            return {}

    def _validate_snps(
        self, snp_dict: dict[str, dict[str, str | int | float]]
    ) -> dict[str, dict[str, str | int | float]]:
        """Validate SNPs through dbSNP.

        Parameters
        ----------
        snp_dict : Dict[str, Dict[str, Any]]
            SNP dictionary to validate.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Validated SNP dictionary.
        """
        if not self._snp_db:
            return snp_dict

        validated_dict: dict[str, dict[str, str | int | float]] = {}

        for rsid, info in snp_dict.items():
            validation = self._snp_db.validate_rsid(rsid)

            if validation.exists and validation.validated:
                # Use validated data
                validated_dict[rsid] = {
                    "gene": validation.gene or info.get("gene", ""),
                    "chromosome": validation.chromosome or info.get("chromosome", ""),
                    "position": validation.position or info.get("position", 0),
                    "trait": info.get("trait", ""),
                    "validated": True,
                }
            else:
                # Keep original but mark as not validated
                validated_dict[rsid] = {**info, "validated": False}

        return validated_dict

    def _interpret_enhanced(
        self,
        df: DataFrame,
        question: str,
        snp_dict: dict[str, dict[str, str | int | float]],
    ) -> InterpretationResult:
        """Create enhanced interpretation with structured output.

        Parameters
        ----------
        df : DataFrame
            Filtered SNP data.
        question : str
            User's question.
        snp_dict : Dict[str, Dict[str, Any]]
            SNP metadata dictionary.

        Returns
        -------
        InterpretationResult
            Structured interpretation result.
        """
        data_str = df.to_string(index=False)

        prompt = f"""The following SNP data was extracted from a user's DNA file
based on the question: {question}

{data_str}

Provide:
1. A concise interpretation (2-3 sentences)
2. A confidence score (0.0 to 1.0) based on scientific evidence
3. Important caveats or disclaimers

Format your response as:
INTERPRETATION: [your interpretation]
CONFIDENCE: [0.0-1.0]
CAVEATS: [caveat 1]; [caveat 2]; ...
"""

        try:
            response = self.llm([HumanMessage(content=prompt)]).content.strip()

            # Parse structured response
            interpretation = self._parse_interpretation(response)
            confidence = self._parse_confidence(response)
            caveats = self._parse_caveats(response)

            # Build SNP results
            snp_results = []
            for _, row in df.iterrows():
                rsid = str(row["RSID"])
                snp_info = snp_dict.get(rsid, {})

                snp_results.append(
                    SNPResult(
                        rsid=rsid,
                        genotype=str(row["GENOTYPE"]),
                        gene=str(snp_info.get("gene", "")),
                        trait=str(snp_info.get("trait", "")),
                        validated=bool(snp_info.get("validated", False)),
                        similarity=float(snp_info.get("similarity", 0.0)),
                    )
                )

            # Add standard disclaimer
            if "This is not medical advice" not in caveats:
                caveats.insert(
                    0,
                    "This is not medical advice. Consult with a healthcare "
                    "professional for medical decisions.",
                )

            return InterpretationResult(
                question=question,
                interpretation=interpretation,
                confidence=confidence,
                snps_found=snp_results,
                sources=["dbSNP", "Vector database"],
                caveats=caveats,
            )

        except Exception as exc:
            logger.error(f"Failed to create enhanced interpretation: {exc}")
            return InterpretationResult(
                question=question,
                interpretation=data_str,
                confidence=0.5,
                caveats=["Error parsing LLM response"],
            )

    @staticmethod
    def _parse_interpretation(response: str) -> str:
        """Parse interpretation from LLM response."""
        for line in response.split("\n"):
            if line.startswith("INTERPRETATION:"):
                return line.replace("INTERPRETATION:", "").strip()
        return response.split("CONFIDENCE:")[0].strip()

    @staticmethod
    def _parse_confidence(response: str) -> float:
        """Parse confidence score from LLM response."""
        for line in response.split("\n"):
            if line.startswith("CONFIDENCE:"):
                try:
                    score = float(line.replace("CONFIDENCE:", "").strip().split()[0])
                    return max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
        return 0.5

    @staticmethod
    def _parse_caveats(response: str) -> list[str]:
        """Parse caveats from LLM response."""
        for line in response.split("\n"):
            if line.startswith("CAVEATS:"):
                caveat_str = line.replace("CAVEATS:", "").strip()
                return [c.strip() for c in caveat_str.split(";") if c.strip()]
        return []

    # Reuse methods from original ChatDNA
    @staticmethod
    def _hash_file(dna_file: Path) -> str:
        """Return a hash of dna_file to detect content changes."""
        return hashlib.sha256(dna_file.read_bytes()).hexdigest()

    def _load_dna(self, dna_file: Path, file_hash: str) -> DataFrame:
        """Load dna_file into a DataFrame with caching."""
        if not dna_file.exists():
            raise FileNotFoundError(f"DNA file '{dna_file}' does not exist.")

        if dna_file.suffix.lower() not in {".csv", ".txt", ".gz", ".zip"}:
            raise ValueError(
                f"Unsupported DNA file extension: '{dna_file.suffix}'. "
                "Expected one of .csv, .txt, .gz or .zip."
            )

        cached = self._dna_cache.get(dna_file)
        if cached is None or cached[0] != file_hash:
            read_csv_kwargs = {
                "comment": "#",
                "delimiter": ",",
                "names": ["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
                "compression": "infer",
            }
            if dna_file.stat().st_size > 50 * 1024 * 1024:
                chunks = pd.read_csv(dna_file, chunksize=100_000, **read_csv_kwargs)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(dna_file, **read_csv_kwargs)
            self._dna_cache[dna_file] = (file_hash, df)
        return self._dna_cache[dna_file][1]

    def _filter_snps(
        self, df: DataFrame, snp_dict: dict[str, dict[str, str | int | float]]
    ) -> DataFrame:
        """Return rows from df whose RSIDs appear in snp_dict."""
        if not snp_dict:
            return DataFrame()

        matched = df[df["RSID"].isin(snp_dict.keys())].copy()
        if matched.empty:
            return matched

        details = (
            pd.DataFrame.from_dict(snp_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "RSID"})
        )
        return matched.merge(details, on="RSID", how="left")
