"""Response schemas for the API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from dna_rag.models import SNPResult

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class LLMModelsInfo(BaseModel):
    """Models used for each pipeline step."""

    snp_identification: str
    interpretation: str


class AnalysisResponse(BaseModel):
    """Response from ``POST /api/v1/analyze``."""

    id: str
    question: str
    matched_snps: list[SNPResult]
    interpretation: str
    snp_count_requested: int
    snp_count_matched: int
    cached: bool = False
    processing_time_ms: int = 0
    llm_models: LLMModelsInfo | None = None


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------


class FileUploadResponse(BaseModel):
    """Returned after a successful DNA file upload."""

    file_id: str
    filename: str
    format: str
    row_count: int
    size_bytes: int
    uploaded_at: datetime


class FileMetaResponse(BaseModel):
    """Metadata for a previously uploaded file."""

    file_id: str
    filename: str
    format: str
    row_count: int
    size_bytes: int
    uploaded_at: datetime


# ---------------------------------------------------------------------------
# Jobs (async analysis)
# ---------------------------------------------------------------------------


class JobCreateResponse(BaseModel):
    """Returned when an async analysis job is submitted."""

    job_id: str
    status: Literal["pending"] = "pending"
    poll_url: str
    websocket_url: str


class JobStatusResponse(BaseModel):
    """Returned when polling a job's status."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    completed_at: datetime | None = None
    result: AnalysisResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Machine-readable error detail (RFC 7807 inspired)."""

    code: str
    message: str
    details: dict | None = None  # type: ignore[type-arg]


class ErrorResponse(BaseModel):
    """Envelope for error responses."""

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness / readiness response."""

    status: Literal["ok", "degraded", "error"]
    version: str
    checks: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


class FormatsResponse(BaseModel):
    """Supported DNA file formats."""

    formats: list[str]


# ---------------------------------------------------------------------------
# WebSocket progress
# ---------------------------------------------------------------------------


class ProgressEvent(BaseModel):
    """Server-to-client WebSocket progress message."""

    step: Literal[
        "parsing",
        "snp_identification",
        "filtering",
        "interpretation",
        "complete",
        "error",
    ]
    status: Literal["started", "done", "streaming", "error"]
    data: dict | None = None  # type: ignore[type-arg]
