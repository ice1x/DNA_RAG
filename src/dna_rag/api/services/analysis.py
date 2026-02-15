"""Analysis service -- orchestrates the core engine with file lookups.

This thin layer translates between the HTTP world (file IDs, multipart
uploads, temporary paths) and the core :class:`~dna_rag.engine.DNAAnalysisEngine`.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import BinaryIO

from dna_rag.api.config import APISettings
from dna_rag.api.schemas.responses import AnalysisResponse, LLMModelsInfo
from dna_rag.api.services.files import FileService
from dna_rag.engine import DNAAnalysisEngine
from dna_rag.logging import get_logger
from dna_rag.models import AnalysisResult

logger = get_logger(__name__)


class AnalysisService:
    """Coordinates file resolution and engine invocation.

    Args:
        engine: The core DNA analysis engine.
        file_service: File storage manager.
        settings: API settings (for LLM model names in response).
    """

    def __init__(
        self,
        engine: DNAAnalysisEngine,
        file_service: FileService,
        settings: APISettings,
    ) -> None:
        self._engine = engine
        self._file_service = file_service
        self._settings = settings

    @property
    def engine(self) -> DNAAnalysisEngine:
        """Expose the underlying engine for direct use (e.g. WebSocket)."""
        return self._engine

    @property
    def file_service(self) -> FileService:
        """Expose file service for direct lookups."""
        return self._file_service

    def analyze_with_file_id(
        self, question: str, file_id: str,
    ) -> AnalysisResponse:
        """Run analysis on a previously uploaded file."""
        path = self._file_service.get_path(file_id)
        if path is None:
            raise FileNotFoundError(f"File not found: {file_id}")
        return self._run(question, path)

    def analyze_with_upload(
        self, question: str, filename: str, data: BinaryIO,
    ) -> AnalysisResponse:
        """Save an uploaded file and run analysis on it."""
        meta = self._file_service.save(filename, data)
        path = Path(meta.disk_path)
        return self._run(question, path)

    # ------------------------------------------------------------------

    def _run(self, question: str, dna_file: Path) -> AnalysisResponse:
        """Execute the core engine and build the API response."""
        start = time.monotonic()
        result: AnalysisResult = self._engine.analyze(question, dna_file)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        return AnalysisResponse(
            id=f"ana_{uuid.uuid4().hex[:12]}",
            question=result.question,
            matched_snps=result.matched_snps,
            interpretation=result.interpretation,
            snp_count_requested=result.snp_count_requested,
            snp_count_matched=result.snp_count_matched,
            cached=result.cached,
            processing_time_ms=elapsed_ms,
            llm_models=LLMModelsInfo(
                snp_identification=self._settings.llm_model,
                interpretation=(
                    self._settings.llm_interp_model
                    or self._settings.llm_model
                ),
            ),
        )
