"""Request schemas for the API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for ``POST /api/v1/analyze`` (JSON variant)."""

    question: str = Field(..., min_length=1, max_length=2000)
    file_id: str | None = Field(
        None, description="Reference to a previously uploaded DNA file.",
    )


class WSAnalyzeRequest(BaseModel):
    """Message sent from client over the analyze WebSocket."""

    question: str = Field(..., min_length=1, max_length=2000)
    file_id: str = Field(..., description="Previously uploaded file ID.")
