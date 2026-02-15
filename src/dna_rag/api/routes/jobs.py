"""Job status polling endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from dna_rag.api.dependencies import get_job_store
from dna_rag.api.schemas.responses import AnalysisResponse, JobStatusResponse
from dna_rag.api.services.jobs import JobStore

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    job_store: Annotated[JobStore, Depends(get_job_store)],
) -> JobStatusResponse:
    """Poll the status of an async analysis job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result = None
    if job.result is not None:
        result = AnalysisResponse(**job.result)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        result=result,
        error=job.error,
    )
