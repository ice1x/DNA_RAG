"""FastAPI application factory for DNA RAG.

Usage::

    # Development
    uvicorn dna_rag.api.main:app --reload

    # Or via the entry point
    dna-rag-api
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dna_rag._version import __version__
from dna_rag.api.config import APISettings
from dna_rag.api.dependencies import get_settings
from dna_rag.api.middleware.auth import AuthMiddleware
from dna_rag.api.middleware.rate_limit import RateLimitMiddleware
from dna_rag.api.middleware.request_id import RequestIdMiddleware
from dna_rag.api.routes import analyze, files, health, jobs, websocket
from dna_rag.exceptions import (
    AnalysisError,
    DNARagError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    ParsingError,
)
from dna_rag.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception → HTTP status mapping
# ---------------------------------------------------------------------------

_EXCEPTION_MAP: dict[type[DNARagError], tuple[int, str]] = {
    AnalysisError: (422, "ANALYSIS_ERROR"),
    ParsingError: (400, "PARSING_ERROR"),
    LLMConnectionError: (502, "LLM_UNAVAILABLE"),
    LLMRateLimitError: (503, "LLM_RATE_LIMITED"),
    LLMResponseError: (502, "LLM_RESPONSE_ERROR"),
}


def _map_exception(exc: DNARagError) -> tuple[int, str]:
    """Return (status_code, error_code) for a core exception."""
    for exc_type, mapping in _EXCEPTION_MAP.items():
        if isinstance(exc, exc_type):
            return mapping
    return 500, "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown hook."""
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)
    logger.info(
        "api_startup",
        version=__version__,
        host=settings.api_host,
        port=settings.api_port,
    )
    yield
    logger.info("api_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        settings: Optional override (useful for testing).
            If ``None``, settings are loaded from the environment.
    """
    if settings is not None:
        # Override the cached singleton so DI picks up the custom settings
        get_settings.cache_clear()
        from dna_rag.api import dependencies

        dependencies.get_settings = lambda: settings  # type: ignore[assignment]
        # Also clear dependent caches
        dependencies.get_engine.cache_clear()
        dependencies.get_file_service.cache_clear()
        dependencies.get_job_store.cache_clear()
        dependencies.get_analysis_service.cache_clear()

    _settings = settings or get_settings()

    application = FastAPI(
        title="DNA RAG API",
        description="Genetic analysis pipeline powered by LLMs",
        version=__version__,
        lifespan=lifespan,
    )

    # --- Middleware (outermost first) ---
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestIdMiddleware)
    application.add_middleware(
        RateLimitMiddleware,
        max_requests=_settings.rate_limit_per_minute,
        window_seconds=60,
    )
    application.add_middleware(
        AuthMiddleware,
        api_keys=set(_settings.api_keys) if _settings.api_keys else None,
        enabled=_settings.auth_enabled,
    )

    # --- Exception handlers ---
    @application.exception_handler(DNARagError)
    async def dna_rag_error_handler(
        request: Request, exc: DNARagError,
    ) -> JSONResponse:
        status, code = _map_exception(exc)
        return JSONResponse(
            status_code=status,
            content={
                "error": {"code": code, "message": str(exc)},
            },
        )

    # --- Routes ---
    application.include_router(health.router)
    application.include_router(analyze.router)
    application.include_router(files.router)
    application.include_router(jobs.router)
    application.include_router(websocket.router)

    return application


def _get_app() -> FastAPI:
    """Lazy app factory for ``uvicorn dna_rag.api.main:app``.

    This avoids triggering settings validation at import time (which
    fails in test environments that haven't set ``DNA_RAG_LLM_API_KEY``).
    """
    return create_app()


# Uvicorn expects a callable or an ASGI instance.  We expose the factory
# so that ``uvicorn dna_rag.api.main:app`` works (uvicorn detects the
# factory pattern when ``--factory`` is used, or we can use a lazy proxy).
app: FastAPI | None = None  # populated by ``run()`` or by tests via ``create_app()``


def run() -> None:  # pragma: no cover
    """Entry point for ``dna-rag-api`` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "dna_rag.api.main:_get_app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        factory=True,
    )
