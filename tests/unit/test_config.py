"""Tests for application configuration."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from dna_rag.config import Settings


class TestSettingsDefaults:
    """Default values and basic loading."""

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-test")
        s = Settings()  # type: ignore[call-arg]
        assert s.llm_api_key.get_secret_value() == "sk-test"
        assert s.llm_provider == "deepseek"
        assert s.llm_model == "deepseek-r1:free"

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-test")
        s = Settings()  # type: ignore[call-arg]
        assert s.llm_temperature == 0.0
        assert s.llm_max_retries == 3
        assert s.llm_timeout == 60.0
        assert s.cache_backend == "memory"
        assert s.cache_max_size == 1000
        assert s.log_level == "INFO"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("DNA_RAG_LLM_API_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]


class TestSettingsValidation:
    """Pydantic validation rules."""

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            Settings(llm_api_key="k", llm_temperature=3.0)  # type: ignore[arg-type]

    def test_max_retries_out_of_range(self):
        with pytest.raises(ValidationError):
            Settings(llm_api_key="k", llm_max_retries=20)  # type: ignore[arg-type]

    def test_secret_str_not_exposed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "super-secret")
        s = Settings()  # type: ignore[call-arg]
        assert "super-secret" not in repr(s)
        assert "super-secret" not in str(s.llm_api_key)
        assert isinstance(s.llm_api_key, SecretStr)


class TestInterpretationLLMSettings:
    """Per-step LLM configuration."""

    def test_no_interp_llm_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-test")
        s = Settings()  # type: ignore[call-arg]
        assert not s.has_separate_interp_llm
        assert s.llm_interp_provider is None

    def test_has_separate_interp_llm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-main")
        s = Settings(
            llm_api_key="sk-main",  # type: ignore[arg-type]
            llm_interp_provider="openai_compat",
            llm_interp_api_key="sk-interp",  # type: ignore[arg-type]
            llm_interp_model="gpt-4o-mini",
            llm_interp_base_url="https://api.openai.com/v1",
        )
        assert s.has_separate_interp_llm

    def test_get_interp_settings_as_primary(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-main")
        s = Settings(
            llm_api_key="sk-main",  # type: ignore[arg-type]
            llm_interp_provider="openai_compat",
            llm_interp_api_key="sk-interp",  # type: ignore[arg-type]
            llm_interp_model="gpt-4o-mini",
            llm_interp_base_url="https://api.openai.com/v1",
        )
        interp = s.get_interp_settings_as_primary()
        assert interp.llm_provider == "openai_compat"
        assert interp.llm_api_key.get_secret_value() == "sk-interp"
        assert interp.llm_model == "gpt-4o-mini"
        assert interp.llm_base_url == "https://api.openai.com/v1"

    def test_interp_falls_back_to_primary(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-main")
        s = Settings(
            llm_api_key="sk-main",  # type: ignore[arg-type]
            llm_interp_provider="deepseek",
        )
        interp = s.get_interp_settings_as_primary()
        # Should fall back to primary API key and base URL
        assert interp.llm_api_key.get_secret_value() == "sk-main"
        assert interp.llm_base_url == "https://api.deepseek.com/v1"

    def test_interp_preserves_rag_validation_settings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "sk-main")
        s = Settings(
            llm_api_key="sk-main",  # type: ignore[arg-type]
            llm_interp_provider="deepseek",
            rag_enabled=True,
            rag_search_results=5,
            validation_enabled=True,
        )
        interp = s.get_interp_settings_as_primary()
        assert interp.rag_enabled is True
        assert interp.rag_search_results == 5
        assert interp.validation_enabled is True


class TestRAGSettings:
    """RAG and validation configuration."""

    def test_rag_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test")
        s = Settings()  # type: ignore[call-arg]
        assert s.rag_enabled is True
        assert s.rag_persist_directory is None
        assert s.rag_embedding_model == "all-MiniLM-L6-v2"
        assert s.rag_collection_name == "snp_traits"
        assert s.rag_search_results == 10
        assert s.rag_min_similarity == 0.3

    def test_rag_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test")
        monkeypatch.setenv("DNA_RAG_RAG_ENABLED", "true")
        monkeypatch.setenv("DNA_RAG_RAG_PERSIST_DIRECTORY", "/tmp/chroma")
        s = Settings()  # type: ignore[call-arg]
        assert s.rag_enabled is True
        assert s.rag_persist_directory == "/tmp/chroma"

    def test_validation_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test")
        s = Settings()  # type: ignore[call-arg]
        assert s.validation_enabled is False
        assert s.validation_timeout == 10.0
        assert s.validation_rate_limit_delay == 0.34

    def test_validation_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test")
        monkeypatch.setenv("DNA_RAG_VALIDATION_ENABLED", "true")
        s = Settings()  # type: ignore[call-arg]
        assert s.validation_enabled is True

    def test_rag_search_results_validation(self):
        with pytest.raises(ValidationError):
            Settings(llm_api_key="k", rag_search_results=0)  # type: ignore[arg-type]

    def test_rag_min_similarity_validation(self):
        with pytest.raises(ValidationError):
            Settings(llm_api_key="k", rag_min_similarity=-0.1)  # type: ignore[arg-type]
