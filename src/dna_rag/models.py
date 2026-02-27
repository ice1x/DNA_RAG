"""Pydantic data models for dna-rag.

These models define the contracts between components:

* :class:`SNPMetadata` -- metadata about a single SNP as returned by the LLM.
* :class:`SNPDictResponse` -- validated first-step LLM response.
* :class:`SNPResult` -- a matched SNP from the user's DNA file.
* :class:`AnalysisResult` -- the complete pipeline result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SNPMetadata(BaseModel):
    """Metadata about a single SNP as identified by the LLM."""

    gene: str
    chromosome: str
    position: int = Field(ge=0)
    trait: str

    @field_validator("chromosome")
    @classmethod
    def normalise_chromosome(cls, v: str) -> str:
        """Normalise chromosome identifiers.

        Accepts ``"1"``--``"22"``, ``"X"``, ``"Y"``, ``"MT"`` (or ``"M"``).
        """
        v = str(v).strip().upper()
        if v.isdigit() and 1 <= int(v) <= 22:
            return v
        if v in ("X", "Y", "MT"):
            return v
        if v == "M":
            return "MT"
        raise ValueError(f"Invalid chromosome: {v!r}")


class SNPDictResponse(BaseModel):
    """Validated response from the first LLM call (SNP identification).

    Keys that do not look like valid RSID identifiers (i.e. do not start
    with ``"rs"``) are silently dropped during validation.
    """

    snps: dict[str, SNPMetadata] = Field(default_factory=dict)

    @field_validator("snps")
    @classmethod
    def validate_rsid_keys(
        cls, v: dict[str, SNPMetadata],
    ) -> dict[str, SNPMetadata]:
        """Keep only entries whose key starts with ``rs``."""
        return {
            k.strip().lower(): meta
            for k, meta in v.items()
            if k.strip().lower().startswith("rs")
        }


class SNPResult(BaseModel):
    """A single SNP from the user's DNA file matched against LLM metadata."""

    rsid: str
    chromosome: str
    position: int
    genotype: str
    gene: str = "unknown"
    trait: str = "unknown"


class AnalysisResult(BaseModel):
    """Complete result returned by :meth:`DNAAnalysisEngine.analyze`."""

    question: str
    matched_snps: list[SNPResult]
    interpretation: str
    snp_count_requested: int = Field(
        description="Number of SNPs the LLM identified as relevant",
    )
    snp_count_matched: int = Field(
        description="Number of those SNPs found in the user's DNA file",
    )
    cached: bool = False
    rag_context_used: bool = False
    validation_used: bool = False
