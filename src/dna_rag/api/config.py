"""API-specific configuration extending the core Settings.

All values can be set via environment variables with the ``DNA_RAG_`` prefix.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from dna_rag.config import Settings


class APISettings(Settings):
    """Configuration for the FastAPI service layer.

    Inherits all core settings (LLM, cache, logging, parser) and adds
    server, auth, file-storage, and rate-limit options.
    """

    model_config = SettingsConfigDict(
        env_prefix="DNA_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server --------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Auth ----------------------------------------------------------------
    auth_enabled: bool = False
    api_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Comma-separated list of valid API keys. "
            "Set DNA_RAG_API_KEYS='key1,key2' to enable."
        ),
    )

    # --- File storage --------------------------------------------------------
    upload_dir: str = "/tmp/dna_rag_uploads"
    file_max_size_mb: int = Field(50, ge=1, le=200)
    file_retention_hours: int = Field(24, ge=1)

    # --- Rate limiting -------------------------------------------------------
    rate_limit_per_minute: int = Field(60, ge=1)

    # --- Jobs ----------------------------------------------------------------
    job_ttl_seconds: int = Field(3600, ge=60)
