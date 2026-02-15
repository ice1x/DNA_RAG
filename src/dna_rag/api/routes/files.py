"""File upload and management endpoints."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from dna_rag.api.dependencies import get_file_service
from dna_rag.api.schemas.responses import FileMetaResponse, FileUploadResponse
from dna_rag.api.services.files import FileService

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.post("", response_model=FileUploadResponse, status_code=201)
async def upload_file(
    file: UploadFile,
    file_service: Annotated[FileService, Depends(get_file_service)],
) -> FileUploadResponse:
    """Upload a DNA data file for subsequent analysis."""
    try:
        meta = await asyncio.to_thread(
            file_service.save,
            file.filename or "upload.txt",
            file.file,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileUploadResponse(
        file_id=meta.file_id,
        filename=meta.filename,
        format=meta.format,
        row_count=meta.row_count,
        size_bytes=meta.size_bytes,
        uploaded_at=meta.uploaded_at,
    )


@router.get("/{file_id}", response_model=FileMetaResponse)
async def get_file_meta(
    file_id: str,
    file_service: Annotated[FileService, Depends(get_file_service)],
) -> FileMetaResponse:
    """Return metadata for a previously uploaded file."""
    meta = file_service.get_meta(file_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return FileMetaResponse(
        file_id=meta.file_id,
        filename=meta.filename,
        format=meta.format,
        row_count=meta.row_count,
        size_bytes=meta.size_bytes,
        uploaded_at=meta.uploaded_at,
    )


@router.delete("/{file_id}", status_code=204)
async def delete_file(
    file_id: str,
    file_service: Annotated[FileService, Depends(get_file_service)],
) -> None:
    """Delete an uploaded DNA file."""
    if not file_service.delete(file_id):
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
