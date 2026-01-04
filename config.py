"""Configuration management for DNA_RAG.

This module provides centralized configuration management using environment
variables. It uses pydantic-settings for validation and python-dotenv for
loading .env files.

Environment Variables:
    DEEPSEEK_API_KEY: DeepSeek API key (required)
    DNA_RAG_CACHE_TTL: SNP database cache TTL in seconds (default: 3600)
    DNA_RAG_MAX_CACHE_SIZE: Maximum cache size (default: 1000)
    DNA_RAG_REQUEST_TIMEOUT: HTTP request timeout in seconds (default: 10.0)
    DNA_RAG_VECTOR_STORE_PATH: Path to vector store directory (default: ./data/vector_store)
    DNA_RAG_EMBEDDING_MODEL: Sentence transformer model name (default: all-MiniLM-L6-v2)
    DNA_RAG_USE_VECTOR_STORE: Enable vector store (default: true)
    DNA_RAG_USE_VALIDATION: Enable SNP validation (default: true)
    DNA_RAG_LLM_MODEL: LLM model name (default: deepseek-r1:free)
    DNA_RAG_LLM_TEMPERATURE: LLM temperature (default: 0.0)
    DNA_RAG_LLM_MAX_RETRIES: LLM max retries (default: 2)
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DNARAGSettings(BaseSettings):
    """DNA_RAG configuration settings.

    All settings can be overridden via environment variables with the
    DNA_RAG_ prefix or via .env file.
    """

    # API Keys
    deepseek_api_key: str = Field(
        default="",
        alias="DEEPSEEK_API_KEY",
        description="DeepSeek API key",
    )

    # SNP Database Settings
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="SNP database cache TTL in seconds",
    )
    max_cache_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of SNPs to cache",
    )
    request_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="HTTP request timeout in seconds",
    )

    # Vector Store Settings
    vector_store_path: Path | None = Field(
        default=Path("./data/vector_store"),
        description="Path to vector store directory",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name",
    )
    use_vector_store: bool = Field(
        default=True,
        description="Enable vector store for RAG",
    )

    # Validation Settings
    use_validation: bool = Field(
        default=True,
        description="Enable SNP validation through dbSNP",
    )

    # LLM Settings
    llm_model: str = Field(
        default="deepseek-r1:free",
        description="LLM model name",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        description="LLM maximum retries",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DNA_RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("vector_store_path", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path | None) -> Path | None:
        """Parse path from string."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    def validate_api_key(self) -> None:
        """Validate that API key is set.

        Raises
        ------
        ValueError
            If API key is not configured.
        """
        if not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Please set it in your environment or .env file."
            )

    def get_vector_store_path(self) -> Path | None:
        """Get vector store path, creating directory if needed.

        Returns
        -------
        Path | None
            Vector store path or None if not configured.
        """
        if self.vector_store_path:
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            return self.vector_store_path
        return None


# Global settings instance
_settings: DNARAGSettings | None = None


def get_settings() -> DNARAGSettings:
    """Get global settings instance.

    This function returns a singleton settings instance. The instance
    is created on first call and reused for subsequent calls.

    Returns
    -------
    DNARAGSettings
        Global settings instance.

    Examples
    --------
    >>> settings = get_settings()
    >>> settings.validate_api_key()
    >>> print(settings.llm_model)
    """
    global _settings
    if _settings is None:
        _settings = DNARAGSettings()
    return _settings


def reload_settings() -> DNARAGSettings:
    """Reload settings from environment.

    Useful for testing or when environment variables change.

    Returns
    -------
    DNARAGSettings
        New settings instance.
    """
    global _settings
    # Apply backward compatibility before reloading
    if "API_KEY" in os.environ and "DEEPSEEK_API_KEY" not in os.environ:
        os.environ["DEEPSEEK_API_KEY"] = os.environ["API_KEY"]
    _settings = DNARAGSettings()
    return _settings


# Backward compatibility: support old API_KEY env var
def _load_api_key_compat() -> str:
    """Load API key with backward compatibility.

    Checks both DEEPSEEK_API_KEY and legacy API_KEY.

    Returns
    -------
    str
        API key if found, empty string otherwise.
    """
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY", "")


# Update environment if using legacy API_KEY
if "API_KEY" in os.environ and "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = os.environ["API_KEY"]
