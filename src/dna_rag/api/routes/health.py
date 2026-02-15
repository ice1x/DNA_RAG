"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from dna_rag._version import __version__
from dna_rag.api.schemas.responses import FormatsResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe -- always returns 200 if the process is up."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/ready", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Readiness probe -- checks that critical dependencies are available."""
    # PoC: just check that settings and engine can be imported
    checks: dict[str, bool] = {}
    try:
        from dna_rag.api.dependencies import get_engine

        get_engine()
        checks["engine"] = True
    except Exception:
        checks["engine"] = False

    resolved: str = "ok" if all(checks.values()) else "degraded"
    return HealthResponse(status=resolved, version=__version__, checks=checks)  # type: ignore[arg-type]


@router.get("/api/v1/formats", response_model=FormatsResponse)
async def supported_formats() -> FormatsResponse:
    """Return the list of supported DNA file formats."""
    return FormatsResponse(formats=["23andme", "ancestrydna", "myheritage"])
