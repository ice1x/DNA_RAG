"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config import DNARAGSettings, get_settings, reload_settings


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for testing."""
    # Remove all DNA_RAG related env vars
    env_vars = list(os.environ.keys())
    for key in env_vars:
        if key.startswith("DNA_RAG_") or key == "DEEPSEEK_API_KEY" or key == "API_KEY":
            monkeypatch.delenv(key, raising=False)


def test_settings_defaults(clean_env):
    """Test default settings values."""
    settings = DNARAGSettings()

    assert settings.cache_ttl == 3600
    assert settings.max_cache_size == 1000
    assert settings.request_timeout == 10.0
    assert settings.vector_store_path == Path("./data/vector_store")
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.use_vector_store is True
    assert settings.use_validation is True
    assert settings.llm_model == "deepseek-r1:free"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_retries == 2


def test_settings_from_env(clean_env, monkeypatch):
    """Test loading settings from environment variables."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_key")
    monkeypatch.setenv("DNA_RAG_CACHE_TTL", "7200")
    monkeypatch.setenv("DNA_RAG_MAX_CACHE_SIZE", "500")
    monkeypatch.setenv("DNA_RAG_LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("DNA_RAG_USE_VECTOR_STORE", "false")

    settings = DNARAGSettings()

    assert settings.deepseek_api_key == "test_key"
    assert settings.cache_ttl == 7200
    assert settings.max_cache_size == 500
    assert settings.llm_temperature == 0.5
    assert settings.use_vector_store is False


def test_validate_api_key_success(clean_env, monkeypatch):
    """Test API key validation success."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_key")
    settings = DNARAGSettings()

    # Should not raise
    settings.validate_api_key()


def test_validate_api_key_missing(clean_env):
    """Test API key validation failure."""
    settings = DNARAGSettings()

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        settings.validate_api_key()


def test_get_vector_store_path(clean_env, tmp_path, monkeypatch):
    """Test vector store path creation."""
    test_path = tmp_path / "test_vector_store"
    monkeypatch.setenv("DNA_RAG_VECTOR_STORE_PATH", str(test_path))

    settings = DNARAGSettings()
    path = settings.get_vector_store_path()

    assert path == test_path
    assert path.exists()


def test_get_vector_store_path_none(clean_env, monkeypatch):
    """Test vector store path when set to empty."""
    monkeypatch.setenv("DNA_RAG_VECTOR_STORE_PATH", "")

    settings = DNARAGSettings()
    # Empty string should still create a Path object
    path = settings.get_vector_store_path()
    assert path is not None or path == Path("")


def test_path_validation(clean_env, monkeypatch):
    """Test path parsing from string."""
    monkeypatch.setenv("DNA_RAG_VECTOR_STORE_PATH", "/tmp/test")
    settings = DNARAGSettings()

    assert settings.vector_store_path == Path("/tmp/test")


def test_get_settings_singleton():
    """Test that get_settings returns singleton."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_reload_settings(clean_env, monkeypatch):
    """Test reloading settings."""
    # First load
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key1")
    settings1 = get_settings()

    # Change environment
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key2")

    # Reload
    settings2 = reload_settings()

    assert settings1.deepseek_api_key == "key1"
    assert settings2.deepseek_api_key == "key2"


def test_backward_compatibility_api_key(clean_env, monkeypatch):
    """Test backward compatibility with API_KEY env var."""
    monkeypatch.setenv("API_KEY", "legacy_key")

    # Import config to trigger compatibility code
    import config

    config.reload_settings()

    # Should work with legacy API_KEY
    assert os.environ.get("DEEPSEEK_API_KEY") == "legacy_key"


def test_temperature_validation(clean_env, monkeypatch):
    """Test LLM temperature validation."""
    # Valid temperature
    monkeypatch.setenv("DNA_RAG_LLM_TEMPERATURE", "1.0")
    settings = DNARAGSettings()
    assert settings.llm_temperature == 1.0

    # Invalid temperature (too high) should be handled by Pydantic
    monkeypatch.setenv("DNA_RAG_LLM_TEMPERATURE", "3.0")
    with pytest.raises(Exception):  # Pydantic validation error
        DNARAGSettings()


def test_cache_size_validation(clean_env, monkeypatch):
    """Test cache size validation."""
    # Valid cache size
    monkeypatch.setenv("DNA_RAG_MAX_CACHE_SIZE", "100")
    settings = DNARAGSettings()
    assert settings.max_cache_size == 100

    # Invalid cache size (zero or negative)
    monkeypatch.setenv("DNA_RAG_MAX_CACHE_SIZE", "0")
    with pytest.raises(Exception):  # Pydantic validation error
        DNARAGSettings()


def test_settings_case_insensitive(clean_env, monkeypatch):
    """Test that settings are case insensitive."""
    monkeypatch.setenv("dna_rag_cache_ttl", "1234")
    settings = DNARAGSettings()

    assert settings.cache_ttl == 1234


def test_dotenv_loading(clean_env, tmp_path, monkeypatch):
    """Test loading from .env file."""
    # Create .env file
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=from_dotenv\nDNA_RAG_CACHE_TTL=9999\n")

    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Create settings (should load from .env)
    settings = DNARAGSettings()

    assert settings.deepseek_api_key == "from_dotenv"
    assert settings.cache_ttl == 9999
