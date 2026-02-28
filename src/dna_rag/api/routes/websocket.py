"""WebSocket endpoint for streaming analysis progress.

Sends step-by-step progress events to the client as the engine
executes each phase of the pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from dna_rag.api.dependencies import get_analysis_service
from dna_rag.api.schemas.responses import ProgressEvent
from dna_rag.api.services.analysis import AnalysisService
from dna_rag.exceptions import (
    AnalysisError,
    LLMError,
)
from dna_rag.logging import get_logger
from dna_rag.parsers.detector import detect_and_parse

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/v1/analyze")
async def ws_analyze(
    websocket: WebSocket,
    service: AnalysisService = Depends(get_analysis_service),  # noqa: B008
) -> None:
    """WebSocket endpoint for streaming DNA analysis.

    **Client sends**::

        { "question": "caffeine metabolism", "file_id": "file_abc123" }

    **Server streams**::

        { "step": "parsing", "status": "started" }
        { "step": "parsing", "status": "done", "data": {"rows": 610000} }
        { "step": "snp_identification", "status": "started", "data": {"model": "..."} }
        ...
        { "step": "complete", "status": "done", "data": { ... result ... } }
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
        question = msg.get("question", "")
        file_id = msg.get("file_id", "")

        if not question or not file_id:
            await _send_error(websocket, "Both 'question' and 'file_id' are required.")
            return

        path = service.file_service.get_path(file_id)
        if path is None:
            await _send_error(websocket, f"File not found: {file_id}")
            return

        await _run_streaming(websocket, service, question, path)

    except WebSocketDisconnect:
        logger.info("ws_disconnected")
    except json.JSONDecodeError:
        await _send_error(websocket, "Invalid JSON.")
    except Exception as exc:
        logger.warning("ws_error", error=str(exc))
        await _send_error(websocket, str(exc))
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


async def _run_streaming(
    ws: WebSocket,
    service: AnalysisService,
    question: str,
    dna_file: Path,
) -> None:
    """Execute analysis with progress events sent over *ws*."""
    engine = service.engine
    settings = service._settings

    # --- Step 1: Parse DNA file ---
    await _send(ws, "parsing", "started")
    try:
        df = await asyncio.to_thread(detect_and_parse, dna_file)
    except Exception as exc:
        await _send_error(ws, f"Parsing failed: {exc}")
        return
    await _send(ws, "parsing", "done", {"rows": len(df)})

    # --- Step 2: RAG context + SNP identification ---
    rag_context = await asyncio.to_thread(engine._get_rag_context, question)

    await _send(ws, "snp_identification", "started", {"model": settings.llm_model})
    try:
        snp_response = await asyncio.to_thread(
            engine._get_snp_dict, question, rag_context=rag_context,
        )
    except LLMError as exc:
        await _send_error(ws, f"SNP identification failed: {exc}")
        return
    except AnalysisError as exc:
        await _send_error(ws, str(exc))
        return

    if not snp_response.snps:
        await _send_error(ws, "No relevant SNPs identified for this question.")
        return
    await _send(ws, "snp_identification", "done", {"snps_found": len(snp_response.snps)})

    # --- Step 3: Filtering ---
    await _send(ws, "filtering", "started")
    matched_df = engine._filter_snps(df, snp_response.snps)
    if matched_df.empty:
        await _send_error(ws, "No matching variants found in your DNA file.")
        return
    matched_snps = engine._dataframe_to_snp_results(matched_df)
    await _send(ws, "filtering", "done", {"matched": len(matched_snps)})

    # --- Step 4: Interpretation ---
    interp_model = settings.llm_interp_model or settings.llm_model
    await _send(ws, "interpretation", "started", {"model": interp_model})
    try:
        interpretation = await asyncio.to_thread(
            engine._interpret, matched_df, question,
        )
    except LLMError as exc:
        await _send_error(ws, f"Interpretation failed: {exc}")
        return
    await _send(ws, "interpretation", "done")

    # --- Complete ---
    result = {
        "question": question,
        "interpretation": interpretation,
        "snp_count_requested": len(snp_response.snps),
        "snp_count_matched": len(matched_snps),
        "matched_snps": [s.model_dump() for s in matched_snps],
    }
    await _send(ws, "complete", "done", result)


async def _send(
    ws: WebSocket,
    step: str,
    status: str,
    data: dict | None = None,  # type: ignore[type-arg]
) -> None:
    """Send a progress event."""
    event = ProgressEvent(step=step, status=status, data=data)  # type: ignore[arg-type]
    await ws.send_text(event.model_dump_json())


async def _send_error(ws: WebSocket, message: str) -> None:
    """Send an error event and close."""
    event = ProgressEvent(step="error", status="error", data={"message": message})  # type: ignore[arg-type]
    await ws.send_text(event.model_dump_json())
