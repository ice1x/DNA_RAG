"""FastAPI dependency injection wiring.

Provides singleton-per-process instances of the core engine, settings,
file service, and job store via :func:`functools.lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache

from dna_rag.api.config import APISettings
from dna_rag.api.services.analysis import AnalysisService
from dna_rag.api.services.files import FileService
from dna_rag.api.services.jobs import JobStore
from dna_rag.cache.memory import InMemoryCache
from dna_rag.config import Settings
from dna_rag.engine import DNAAnalysisEngine
from dna_rag.exceptions import ConfigurationError
from dna_rag.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Return the singleton :class:`APISettings` instance."""
    return APISettings()  # type: ignore[call-arg]


def _make_llm_provider(settings: APISettings | Settings):  # noqa: ANN202
    """Create an LLM provider from *settings*."""
    if settings.llm_provider == "deepseek":
        from dna_rag.llm.deepseek import DeepSeekProvider

        return DeepSeekProvider(settings)
    elif settings.llm_provider == "openai_compat":
        from dna_rag.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings)
    else:
        raise ConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")


@lru_cache(maxsize=1)
def get_engine() -> DNAAnalysisEngine:
    """Return the singleton :class:`DNAAnalysisEngine`."""
    settings = get_settings()

    snp_llm = _make_llm_provider(settings)

    interp_llm = None
    if settings.has_separate_interp_llm:
        interp_settings = settings.get_interp_settings_as_primary()
        interp_llm = _make_llm_provider(interp_settings)

    cache = (
        InMemoryCache(
            max_size=settings.cache_max_size,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        if settings.cache_backend == "memory"
        else None
    )

    logger.info(
        "engine_created",
        snp_provider=settings.llm_provider,
        interp_provider=(
            settings.llm_interp_provider or settings.llm_provider
        ),
        cache_backend=settings.cache_backend,
    )
    return DNAAnalysisEngine(
        snp_llm=snp_llm,
        interpretation_llm=interp_llm,
        cache=cache,
    )


@lru_cache(maxsize=1)
def get_file_service() -> FileService:
    """Return the singleton :class:`FileService`."""
    settings = get_settings()
    return FileService(
        upload_dir=settings.upload_dir,
        max_size_mb=settings.file_max_size_mb,
    )


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """Return the singleton in-memory :class:`JobStore`."""
    settings = get_settings()
    return JobStore(ttl_seconds=settings.job_ttl_seconds)


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    """Return the singleton :class:`AnalysisService`."""
    return AnalysisService(
        engine=get_engine(),
        file_service=get_file_service(),
        settings=get_settings(),
    )
