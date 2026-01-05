"""Pydantic models for FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DNAQuestionRequest(BaseModel):
    """Request model for DNA question endpoint."""

    question: str = Field(
        ...,
        description="Question about DNA data",
        min_length=1,
        examples=["Am I lactose intolerant?"],
    )
    dna_data: str = Field(
        ...,
        description="DNA data in CSV format (rsid,chromosome,position,genotype)",
        min_length=1,
    )
    use_rag: bool = Field(
        default=False,
        description="Use RAG-enhanced mode with vector store",
    )


class DNAQuestionResponse(BaseModel):
    """Response model for DNA question endpoint."""

    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="Answer to the question")
    provider: str | None = Field(default=None, description="LLM provider used for the answer")


class EnhancedDNAQuestionResponse(BaseModel):
    """Enhanced response model with structured data."""

    question: str = Field(..., description="The question that was asked")
    interpretation: str = Field(..., description="Interpretation of DNA data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    snps_found: list[SNPResult] = Field(default_factory=list, description="SNPs found in the data")
    sources: list[str] = Field(default_factory=list, description="Source citations")
    caveats: list[str] = Field(
        default_factory=list, description="Important caveats and limitations"
    )
    provider: str | None = Field(default=None, description="LLM provider used for the answer")


class SNPResult(BaseModel):
    """Individual SNP result with metadata."""

    rsid: str = Field(..., description="Reference SNP ID")
    genotype: str = Field(..., description="Genotype (e.g., AA, AG, GG)")
    gene: str | None = Field(default=None, description="Associated gene")
    trait: str | None = Field(default=None, description="Associated trait")
    validated: bool = Field(default=False, description="Whether SNP was validated via dbSNP")
    similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Similarity score from vector store"
    )


class PolygenicScoreRequest(BaseModel):
    """Request model for polygenic score calculation."""

    score_name: str = Field(
        ...,
        description="Name of the polygenic score to calculate",
        examples=["alzheimers_risk", "type2_diabetes_risk"],
    )
    dna_data: str = Field(
        ...,
        description="DNA data in CSV format (rsid,chromosome,position,genotype)",
        min_length=1,
    )


class PolygenicScoreResponse(BaseModel):
    """Response model for polygenic score calculation."""

    score_name: str = Field(..., description="Name of the calculated score")
    score: float = Field(..., description="Calculated polygenic score")
    interpretation: str = Field(..., description="Interpretation of the score")
    percentile: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percentile rank (if available)",
    )


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    providers: dict[str, Any] = Field(default_factory=dict, description="Available LLM providers")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
