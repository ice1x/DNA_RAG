"""Application configuration powered by Pydantic Settings.

All values can be set via environment variables prefixed with ``DNA_RAG_``
or through a ``.env`` file in the working directory.

The engine supports **per-step LLM configuration**: you can use a powerful
reasoning model (e.g. DeepSeek R1) for SNP identification and a faster /
cheaper model for interpretation.

If ``llm_interp_*`` settings are not provided, the engine falls back to
the primary ``llm_*`` settings for both steps.

Example::

    # Primary LLM (used for SNP identification, and as default)
    export DNA_RAG_LLM_API_KEY=sk-...
    export DNA_RAG_LLM_MODEL=deepseek-r1:free

    # Optional: separate LLM for interpretation step
    export DNA_RAG_LLM_INTERP_PROVIDER=openai_compat
    export DNA_RAG_LLM_INTERP_API_KEY=sk-...
    export DNA_RAG_LLM_INTERP_MODEL=gpt-4o-mini
    export DNA_RAG_LLM_INTERP_BASE_URL=https://api.openai.com/v1
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="DNA_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Primary LLM (Step 1: SNP identification + default) ----------------
    llm_provider: Literal["deepseek", "openai_compat"] = "deepseek"
    llm_api_key: SecretStr = Field(
        ..., description="API key for the primary LLM provider",
    )
    llm_model: str = "deepseek-r1:free"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_tokens: int | None = None
    llm_timeout: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(3, ge=0, le=10)
    llm_base_url: str = "https://api.deepseek.com/v1"

    # --- Interpretation LLM (Step 2, optional override) --------------------
    llm_interp_provider: Literal["deepseek", "openai_compat"] | None = None
    llm_interp_api_key: SecretStr | None = None
    llm_interp_model: str | None = None
    llm_interp_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_interp_max_tokens: int | None = None
    llm_interp_timeout: float = Field(60.0, gt=0)
    llm_interp_max_retries: int = Field(3, ge=0, le=10)
    llm_interp_base_url: str | None = None

    # --- Cache -------------------------------------------------------------
    cache_backend: Literal["memory", "none"] = "memory"
    cache_max_size: int = Field(1000, ge=0)
    cache_ttl_seconds: int = Field(3600, ge=0)

    # --- RAG (Vector Store) ------------------------------------------------
    rag_enabled: bool = True
    rag_persist_directory: str | None = None
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_collection_name: str = "snp_traits"
    rag_search_results: int = Field(10, ge=1, le=50)
    rag_min_similarity: float = Field(0.3, ge=0.0, le=1.0)

    # --- SNP Validation (NCBI) --------------------------------------------
    validation_enabled: bool = True
    validation_timeout: float = Field(10.0, gt=0)
    validation_rate_limit_delay: float = Field(0.34, ge=0.0)

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- Parser ------------------------------------------------------------
    default_dna_format: Literal[
        "auto", "23andme", "ancestrydna", "myheritage"
    ] = "auto"

    # --- Disclaimer --------------------------------------------------------
    medical_disclaimer: str = (
        "Disclaimer: This analysis is for educational and research purposes only. "
        "Genetic predisposition is not deterministic — environment, lifestyle, and "
        "other factors play a significant role. Do not make health decisions based "
        "on this output. Consumer DNA tests (23andMe, AncestryDNA, MyHeritage) have "
        "known error rates, especially for insertions, deletions, and CNVs. "
        "You can verify any SNP identifier (e.g. rs12345) by searching it at "
        "https://www.ncbi.nlm.nih.gov/snp/ or https://www.snpedia.com/. "
        "Consult a qualified healthcare provider or genetic counselor "
        "for medical interpretation of genetic data."
    )

    @property
    def has_separate_interp_llm(self) -> bool:
        """Return ``True`` if a separate interpretation LLM is configured."""
        return self.llm_interp_provider is not None

    def get_interp_settings_as_primary(self) -> Settings:
        """Build a :class:`Settings` copy where interp values become primary.

        This is a convenience for creating a second LLM provider using
        the ``llm_interp_*`` settings in the same ``Settings`` shape.
        """
        return Settings(  # type: ignore[call-arg]
            llm_provider=self.llm_interp_provider or self.llm_provider,
            llm_api_key=self.llm_interp_api_key or self.llm_api_key,
            llm_model=self.llm_interp_model or self.llm_model,
            llm_temperature=self.llm_interp_temperature,
            llm_max_tokens=self.llm_interp_max_tokens,
            llm_timeout=self.llm_interp_timeout,
            llm_max_retries=self.llm_interp_max_retries,
            llm_base_url=self.llm_interp_base_url or self.llm_base_url,
            # Copy non-LLM settings as-is
            cache_backend=self.cache_backend,
            cache_max_size=self.cache_max_size,
            cache_ttl_seconds=self.cache_ttl_seconds,
            rag_enabled=self.rag_enabled,
            rag_persist_directory=self.rag_persist_directory,
            rag_embedding_model=self.rag_embedding_model,
            rag_collection_name=self.rag_collection_name,
            rag_search_results=self.rag_search_results,
            rag_min_similarity=self.rag_min_similarity,
            validation_enabled=self.validation_enabled,
            validation_timeout=self.validation_timeout,
            validation_rate_limit_delay=self.validation_rate_limit_delay,
            log_level=self.log_level,
            log_format=self.log_format,
            default_dna_format=self.default_dna_format,
        )
