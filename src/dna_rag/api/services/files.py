"""File upload / storage service.

For the PoC this uses a local filesystem directory.  In production this
would be swapped for S3 or another object store.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel

from dna_rag.logging import get_logger
from dna_rag.parsers.detector import detect_and_parse

logger = get_logger(__name__)


class FileMeta(BaseModel):
    """Metadata for an uploaded DNA file."""

    file_id: str
    filename: str
    format: str
    row_count: int
    size_bytes: int
    sha256: str
    disk_path: str
    uploaded_at: datetime


class FileService:
    """Manages DNA file uploads on the local filesystem.

    Args:
        upload_dir: Directory where uploaded files are stored.
        max_size_mb: Maximum allowed file size in megabytes.
    """

    def __init__(
        self,
        upload_dir: str = "/tmp/dna_rag_uploads",
        max_size_mb: int = 50,
    ) -> None:
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        # In-memory registry  (PoC -- no DB)
        self._files: dict[str, FileMeta] = {}

    def save(self, filename: str, data: BinaryIO) -> FileMeta:
        """Save an uploaded file and return its metadata.

        Raises:
            ValueError: File exceeds the size limit or cannot be parsed.
        """
        content = data.read()
        if len(content) > self._max_size_bytes:
            raise ValueError(
                f"File exceeds {self._max_size_bytes // (1024 * 1024)} MB limit"
            )

        sha = hashlib.sha256(content).hexdigest()

        # Deduplication: if we already have this exact file, reuse it
        for existing in self._files.values():
            if existing.sha256 == sha:
                logger.info("file_deduplicated", file_id=existing.file_id)
                return existing

        file_id = f"file_{uuid.uuid4().hex[:12]}"
        disk_path = self._upload_dir / f"{file_id}_{filename}"
        disk_path.write_bytes(content)

        # Detect format and count rows
        try:
            df = detect_and_parse(disk_path)
            fmt = _detect_format_name(filename, disk_path)
            row_count = len(df)
        except Exception as exc:
            disk_path.unlink(missing_ok=True)
            raise ValueError(f"Cannot parse DNA file: {exc}") from exc

        meta = FileMeta(
            file_id=file_id,
            filename=filename,
            format=fmt,
            row_count=row_count,
            size_bytes=len(content),
            sha256=sha,
            disk_path=str(disk_path),
            uploaded_at=datetime.now(UTC),
        )
        self._files[file_id] = meta
        logger.info(
            "file_uploaded",
            file_id=file_id,
            format=fmt,
            rows=row_count,
            size_bytes=len(content),
        )
        return meta

    def get_meta(self, file_id: str) -> FileMeta | None:
        """Return metadata for *file_id*, or ``None`` if not found."""
        return self._files.get(file_id)

    def get_path(self, file_id: str) -> Path | None:
        """Return the disk path for *file_id*, or ``None``."""
        meta = self._files.get(file_id)
        if meta is None:
            return None
        p = Path(meta.disk_path)
        return p if p.exists() else None

    def delete(self, file_id: str) -> bool:
        """Delete an uploaded file.  Returns ``True`` if found."""
        meta = self._files.pop(file_id, None)
        if meta is None:
            return False
        Path(meta.disk_path).unlink(missing_ok=True)
        logger.info("file_deleted", file_id=file_id)
        return True

    def list_files(self) -> list[FileMeta]:
        """Return all stored file metadata entries."""
        return list(self._files.values())


def _detect_format_name(filename: str, path: Path) -> str:
    """Heuristic format label based on filename and content."""
    name_lower = filename.lower()
    if "ancestry" in name_lower:
        return "ancestrydna"
    if "myheritage" in name_lower:
        return "myheritage"
    if "23andme" in name_lower:
        return "23andme"

    # Fallback: probe parsers
    from dna_rag.parsers.ancestrydna import AncestryDNAParser
    from dna_rag.parsers.myheritage import MyHeritageParser
    from dna_rag.parsers.twentythreeandme import TwentyThreeAndMeParser

    if AncestryDNAParser.can_parse(path):
        return "ancestrydna"
    if TwentyThreeAndMeParser.can_parse(path):
        return "23andme"
    if MyHeritageParser.can_parse(path):
        return "myheritage"
    return "unknown"
