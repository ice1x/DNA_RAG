"""Configuration management for DNA_RAG.

This module provides centralized configuration management using environment
variables. It uses pydantic-settings for validation and python-dotenv for
loading .env files.

Environment Variables:
    # LLM Provider API Keys
    OPENAI_API_KEY: OpenAI API key (optional)
    DEEPSEEK_API_KEY: DeepSeek API key (optional, legacy support)
    DNA_RAG_OPENAI_API_KEY: OpenAI API key (optional)
    DNA_RAG_DEEPSEEK_API_KEY: DeepSeek API key (optional)
    DNA_RAG_LLM_PROVIDERS: Priority list of providers (default: openai,deepseek)

    # Other Settings
    DNA_RAG_CACHE_TTL: SNP database cache TTL in seconds (default: 3600)
    DNA_RAG_MAX_CACHE_SIZE: Maximum cache size (default: 1000)
    DNA_RAG_REQUEST_TIMEOUT: HTTP request timeout in seconds (default: 10.0)
    DNA_RAG_VECTOR_STORE_PATH: Path to vector store directory (default: ./data/vector_store)
    DNA_RAG_EMBEDDING_MODEL: Sentence transformer model name (default: all-MiniLM-L6-v2)
    DNA_RAG_USE_VECTOR_STORE: Enable vector store (default: true)
    DNA_RAG_USE_VALIDATION: Enable SNP validation (default: true)
    DNA_RAG_LLM_TEMPERATURE: LLM temperature (default: 0.0)
    DNA_RAG_LLM_MAX_RETRIES: LLM max retries (default: 2)
    DNA_RAG_OPENAI_MODEL: OpenAI model name (default: gpt-4o-mini)
    DNA_RAG_DEEPSEEK_MODEL: DeepSeek model name (default: deepseek-chat)
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

    # API Keys for LLM Providers
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key",
    )
    deepseek_api_key: str = Field(
        default="",
        description="DeepSeek API key",
    )

    # LLM Provider Priority
    llm_providers: str = Field(
        default="openai,deepseek",
        description="Comma-separated list of LLM providers in priority order",
    )

    # Provider-specific Models
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model name",
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

    @field_validator("openai_api_key", "deepseek_api_key", mode="before")
    @classmethod
    def load_api_key_with_fallback(cls, v: str, info) -> str:
        """Load API key with fallback to non-prefixed env vars."""
        if v:
            return v
        # Fallback to non-prefixed environment variables
        field_name = info.field_name
        if field_name == "openai_api_key":
            return os.environ.get("OPENAI_API_KEY", "")
        elif field_name == "deepseek_api_key":
            # Support legacy DEEPSEEK_API_KEY and API_KEY
            return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY") or ""
        return v

    def validate_api_keys(self) -> None:
        """Validate that at least one API key is configured.

        Raises
        ------
        ValueError
            If no API keys are configured.
        """
        if not self.openai_api_key and not self.deepseek_api_key:
            raise ValueError(
                "No LLM provider API keys configured. "
                "Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in your environment or .env file."
            )

    def get_provider_priority(self) -> list[str]:
        """Get list of providers in priority order.

        Returns
        -------
        list[str]
            Provider names in priority order
        """
        return [p.strip().lower() for p in self.llm_providers.split(",") if p.strip()]

    def create_llm_manager(self):
        """Create LLM provider manager with configured providers.

        Returns
        -------
        LLMProviderManager
            Configured provider manager

        Raises
        ------
        ValueError
            If no valid providers are configured
        """
        from llm_providers import (
            DeepSeekProvider,
            LLMProviderManager,
            OpenAIProvider,
        )

        providers = []
        priority = self.get_provider_priority()

        for provider_name in priority:
            if provider_name == "openai" and self.openai_api_key:
                providers.append(
                    OpenAIProvider(
                        api_key=self.openai_api_key,
                        model=self.openai_model,
                        temperature=self.llm_temperature,
                        max_retries=self.llm_max_retries,
                        timeout=self.request_timeout,
                    )
                )
            elif provider_name == "deepseek" and self.deepseek_api_key:
                providers.append(
                    DeepSeekProvider(
                        api_key=self.deepseek_api_key,
                        model=self.deepseek_model,
                        temperature=self.llm_temperature,
                        max_retries=self.llm_max_retries,
                        timeout=self.request_timeout,
                    )
                )

        return LLMProviderManager(providers)

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
