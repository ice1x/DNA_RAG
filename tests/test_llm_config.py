"""Tests for LLM provider configuration."""

import os

import pytest

from dna_rag.core.config import DNARAGSettings, reload_settings


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Clean LLM-related environment variables."""
    env_vars = list(os.environ.keys())
    for key in env_vars:
        if "API_KEY" in key or key.startswith("DNA_RAG_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()


def test_openai_api_key_loading(clean_llm_env, monkeypatch):
    """Test loading OpenAI API key from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    settings = DNARAGSettings()
    assert settings.openai_api_key == "sk-test123"


def test_deepseek_api_key_loading(clean_llm_env, monkeypatch):
    """Test loading DeepSeek API key from environment."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test456")
    settings = DNARAGSettings()
    assert settings.deepseek_api_key == "ds-test456"


def test_legacy_api_key_support(clean_llm_env, monkeypatch):
    """Test backward compatibility with API_KEY env var."""
    monkeypatch.setenv("API_KEY", "legacy-key")
    settings = DNARAGSettings()
    assert settings.deepseek_api_key == "legacy-key"


def test_prefixed_api_keys(clean_llm_env, monkeypatch):
    """Test loading prefixed API keys."""
    monkeypatch.setenv("DNA_RAG_OPENAI_API_KEY", "openai-prefixed")
    monkeypatch.setenv("DNA_RAG_DEEPSEEK_API_KEY", "deepseek-prefixed")
    settings = DNARAGSettings()
    assert settings.openai_api_key == "openai-prefixed"
    assert settings.deepseek_api_key == "deepseek-prefixed"


def test_provider_priority_default(clean_llm_env):
    """Test default provider priority."""
    settings = DNARAGSettings()
    assert settings.llm_providers == "openai,deepseek"
    assert settings.get_provider_priority() == ["openai", "deepseek"]


def test_provider_priority_custom(clean_llm_env, monkeypatch):
    """Test custom provider priority."""
    monkeypatch.setenv("DNA_RAG_LLM_PROVIDERS", "deepseek,openai")
    settings = DNARAGSettings()
    assert settings.get_provider_priority() == ["deepseek", "openai"]


def test_provider_priority_with_spaces(clean_llm_env, monkeypatch):
    """Test provider priority parsing with spaces."""
    monkeypatch.setenv("DNA_RAG_LLM_PROVIDERS", "openai , deepseek ")
    settings = DNARAGSettings()
    assert settings.get_provider_priority() == ["openai", "deepseek"]


def test_default_models(clean_llm_env):
    """Test default model names."""
    settings = DNARAGSettings()
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.deepseek_model == "deepseek-chat"


def test_custom_models(clean_llm_env, monkeypatch):
    """Test custom model names."""
    monkeypatch.setenv("DNA_RAG_OPENAI_MODEL", "gpt-4")
    monkeypatch.setenv("DNA_RAG_DEEPSEEK_MODEL", "deepseek-coder")
    settings = DNARAGSettings()
    assert settings.openai_model == "gpt-4"
    assert settings.deepseek_model == "deepseek-coder"


def test_validate_api_keys_success(clean_llm_env, monkeypatch):
    """Test API key validation with at least one key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = DNARAGSettings()
    settings.validate_api_keys()  # Should not raise


def test_validate_api_keys_failure(clean_llm_env):
    """Test API key validation fails with no keys."""
    settings = DNARAGSettings()
    with pytest.raises(ValueError, match="No LLM provider API keys configured"):
        settings.validate_api_keys()


def test_create_llm_manager_openai_only(clean_llm_env, monkeypatch):
    """Test creating LLM manager with OpenAI only."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    settings = DNARAGSettings()

    manager = settings.create_llm_manager()
    assert len(manager.providers) == 1
    assert manager.get_active_provider() == "openai"


def test_create_llm_manager_deepseek_only(clean_llm_env, monkeypatch):
    """Test creating LLM manager with DeepSeek only."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    settings = DNARAGSettings()

    manager = settings.create_llm_manager()
    assert len(manager.providers) == 1
    assert manager.get_active_provider() == "deepseek"


def test_create_llm_manager_both_providers(clean_llm_env, monkeypatch):
    """Test creating LLM manager with both providers."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    settings = DNARAGSettings()

    manager = settings.create_llm_manager()
    assert len(manager.providers) == 2
    assert manager.get_active_provider() == "openai"  # First in priority


def test_create_llm_manager_reversed_priority(clean_llm_env, monkeypatch):
    """Test creating LLM manager with reversed priority."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DNA_RAG_LLM_PROVIDERS", "deepseek,openai")
    settings = DNARAGSettings()

    manager = settings.create_llm_manager()
    assert len(manager.providers) == 2
    assert manager.get_active_provider() == "deepseek"  # First in priority


def test_create_llm_manager_no_providers(clean_llm_env):
    """Test creating LLM manager fails with no providers."""
    settings = DNARAGSettings()

    with pytest.raises(ValueError, match="No available LLM providers"):
        settings.create_llm_manager()


def test_llm_settings_applied(clean_llm_env, monkeypatch):
    """Test LLM settings are applied to providers."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DNA_RAG_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("DNA_RAG_LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("DNA_RAG_REQUEST_TIMEOUT", "60.0")
    settings = DNARAGSettings()

    manager = settings.create_llm_manager()
    provider = manager.providers[0]

    assert provider.temperature == 0.7
    assert provider.max_retries == 5
    assert provider.timeout == 60.0
