"""FastAPI dependency injection wiring.

Provides singleton-per-process instances of the core engine, settings,
file service, and job store via :func:`functools.lru_cache`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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


def _build_vector_store(settings: APISettings):  # noqa: ANN202
    """Attempt to create a :class:`SNPVectorStore`.  Returns ``None`` on failure."""
    try:
        from dna_rag.vector_store import SNPVectorStore
    except ImportError:
        logger.warning(
            "rag_unavailable",
            reason="Install with: pip install dna-rag[rag]",
        )
        return None

    persist_dir = Path(settings.rag_persist_directory) if settings.rag_persist_directory else None
    return SNPVectorStore(
        persist_directory=persist_dir,
        embedding_model=settings.rag_embedding_model,
        collection_name=settings.rag_collection_name,
    )


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

    vector_store = None
    if settings.rag_enabled:
        vector_store = _build_vector_store(settings)

    snp_database = None
    if settings.validation_enabled:
        from dna_rag.snp_database import SNPDatabase

        snp_database = SNPDatabase(
            cache=cache,
            request_timeout=settings.validation_timeout,
            rate_limit_delay=settings.validation_rate_limit_delay,
        )

    logger.info(
        "engine_created",
        snp_provider=settings.llm_provider,
        interp_provider=(
            settings.llm_interp_provider or settings.llm_provider
        ),
        cache_backend=settings.cache_backend,
        rag_enabled=vector_store is not None,
        validation_enabled=snp_database is not None,
    )
    return DNAAnalysisEngine(
        snp_llm=snp_llm,
        interpretation_llm=interp_llm,
        cache=cache,
        vector_store=vector_store,
        snp_database=snp_database,
        rag_search_results=settings.rag_search_results,
        rag_min_similarity=settings.rag_min_similarity,
        medical_disclaimer=settings.medical_disclaimer,
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
