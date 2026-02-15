"""Tests for the OpenAI-compatible LLM provider."""

from __future__ import annotations

import pytest

from dna_rag.config import Settings
from dna_rag.exceptions import LLMRateLimitError, LLMResponseError
from dna_rag.llm.openai_compat import OpenAICompatProvider


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_api_key="test-key",
        llm_base_url="https://mock.test/v1",
        llm_model="test-model",
        llm_max_retries=1,
        llm_timeout=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestOpenAICompatInvoke:
    """Tests using pytest-httpx to mock HTTP responses."""

    def test_successful_invocation(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello from OpenAI"}}],
            },
        )
        provider = OpenAICompatProvider(_make_settings())
        result = provider.invoke("hi")
        assert result == "Hello from OpenAI"

    def test_does_not_strip_think_tags(self, httpx_mock):
        """Unlike DeepSeek, the generic provider does NOT strip think tags."""
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "<think>reasoning</think>answer"}},
                ],
            },
        )
        provider = OpenAICompatProvider(_make_settings())
        result = provider.invoke("test")
        assert "<think>" in result

    def test_rate_limit_raises(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            status_code=429,
            text="Rate limited",
        )
        provider = OpenAICompatProvider(_make_settings())
        with pytest.raises(LLMRateLimitError):
            provider.invoke("test")

    def test_server_error_raises(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            status_code=500,
            text="Internal error",
        )
        provider = OpenAICompatProvider(_make_settings())
        with pytest.raises(LLMResponseError):
            provider.invoke("test")

    def test_context_manager(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with OpenAICompatProvider(_make_settings()) as provider:
            result = provider.invoke("test")
        assert result == "ok"
