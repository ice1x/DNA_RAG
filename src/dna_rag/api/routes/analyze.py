"""Analysis endpoints -- synchronous and async."""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from dna_rag.api.dependencies import get_analysis_service, get_job_store
from dna_rag.api.schemas.responses import AnalysisResponse, JobCreateResponse
from dna_rag.api.services.analysis import AnalysisService
from dna_rag.api.services.jobs import JobStore
from dna_rag.exceptions import (
    AnalysisError,
    DNARagError,
    LLMError,
    ParsingError,
)
from dna_rag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


# ---------------------------------------------------------------------------
# POST /api/v1/analyze  (synchronous)
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    question: Annotated[str, Form(...)],
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
    file: Annotated[UploadFile | None, File()] = None,
    file_id: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    """Run a synchronous DNA analysis.

    Provide **either** ``file`` (upload) **or** ``file_id`` (reference to a
    previously uploaded file).
    """
    if file is None and file_id is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' (upload) or 'file_id' must be provided.",
        )

    try:
        if file_id:
            return await asyncio.to_thread(
                service.analyze_with_file_id, question, file_id,
            )
        else:
            assert file is not None
            return await asyncio.to_thread(
                service.analyze_with_upload,
                question,
                file.filename or "upload.txt",
                file.file,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ParsingError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DNARagError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# POST /api/v1/analyze/async  (submit a background job)
# ---------------------------------------------------------------------------


@router.post("/analyze/async", response_model=JobCreateResponse, status_code=202)
async def analyze_async(
    question: Annotated[str, Form(...)],
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
    job_store: Annotated[JobStore, Depends(get_job_store)],
    file: Annotated[UploadFile | None, File()] = None,
    file_id: Annotated[str | None, Form()] = None,
) -> JobCreateResponse:
    """Submit an analysis job for background execution.

    Returns immediately with a ``job_id`` that can be polled via
    ``GET /api/v1/jobs/{job_id}``.
    """
    if file is None and file_id is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' (upload) or 'file_id' must be provided.",
        )

    # If a file was uploaded, save it first to get a file_id
    resolved_file_id = file_id
    if file is not None and file_id is None:
        try:
            meta = await asyncio.to_thread(
                service.file_service.save,
                file.filename or "upload.txt",
                file.file,
            )
            resolved_file_id = meta.file_id
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    assert resolved_file_id is not None
    job = job_store.create(question=question, file_id=resolved_file_id)

    # Launch background thread. We use a plain thread rather than
    # asyncio.create_task because analyze_with_file_id is synchronous
    # and asyncio tasks don't reliably resume under TestClient's event loop.
    thread = threading.Thread(
        target=_run_job,
        args=(job.job_id, question, resolved_file_id, service, job_store),
        daemon=True,
    )
    thread.start()

    return JobCreateResponse(
        job_id=job.job_id,
        poll_url=f"/api/v1/jobs/{job.job_id}",
        websocket_url=f"/ws/v1/jobs/{job.job_id}",
    )


def _run_job(
    job_id: str,
    question: str,
    file_id: str,
    service: AnalysisService,
    job_store: JobStore,
) -> None:
    """Execute an analysis job in a background thread."""
    job_store.set_running(job_id)
    try:
        result = service.analyze_with_file_id(question, file_id)
        job_store.set_completed(job_id, result.model_dump(mode="json"))
    except Exception as exc:
        job_store.set_failed(job_id, str(exc))
        logger.warning("job_failed", job_id=job_id, error=str(exc))
