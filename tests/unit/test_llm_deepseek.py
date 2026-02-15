"""Tests for the DeepSeek LLM provider."""

from __future__ import annotations

import pytest

from dna_rag.config import Settings
from dna_rag.exceptions import LLMRateLimitError, LLMResponseError
from dna_rag.llm.deepseek import DeepSeekProvider


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


class TestDeepSeekInvoke:
    """Tests using pytest-httpx to mock HTTP responses."""

    def test_successful_invocation(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={
                "choices": [{"message": {"content": "Hello world"}}],
            },
        )
        provider = DeepSeekProvider(_make_settings())
        result = provider.invoke("hi")
        assert result == "Hello world"

    def test_strips_think_tags(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>I need to reason about this.</think>"
                                '{"rs1": {"gene": "G"}}'
                            ),
                        }
                    }
                ],
            },
        )
        provider = DeepSeekProvider(_make_settings())
        result = provider.invoke("test")
        assert "<think>" not in result
        assert '{"rs1"' in result

    def test_strips_multiline_think_tags(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>\nLet me think...\nStep 1.\n</think>\n"
                                "The answer is 42."
                            ),
                        }
                    }
                ],
            },
        )
        provider = DeepSeekProvider(_make_settings())
        result = provider.invoke("test")
        assert result == "The answer is 42."

    def test_rate_limit_raises(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            status_code=429,
            text="Rate limited",
        )
        provider = DeepSeekProvider(_make_settings())
        with pytest.raises(LLMRateLimitError):
            provider.invoke("test")

    def test_server_error_raises(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            status_code=500,
            text="Internal error",
        )
        provider = DeepSeekProvider(_make_settings())
        with pytest.raises(LLMResponseError, match="500"):
            provider.invoke("test")

    def test_malformed_response_raises(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={"unexpected": "format"},
        )
        provider = DeepSeekProvider(_make_settings())
        with pytest.raises(LLMResponseError, match="Unexpected"):
            provider.invoke("test")

    def test_sends_correct_payload(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        provider = DeepSeekProvider(_make_settings(llm_temperature=0.5))
        provider.invoke("my prompt")

        request = httpx_mock.get_request()
        import json
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["temperature"] == 0.5
        assert body["messages"][0]["content"] == "my prompt"

    def test_sends_auth_header(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        provider = DeepSeekProvider(_make_settings(llm_api_key="my-secret"))
        provider.invoke("test")

        request = httpx_mock.get_request()
        assert request.headers["authorization"] == "Bearer my-secret"


class TestDeepSeekContextManager:
    """Resource management."""

    def test_context_manager(self, httpx_mock):
        httpx_mock.add_response(
            url="https://mock.test/v1/chat/completions",
            json={"choices": [{"message": {"content": "ok"}}]},
        )
        with DeepSeekProvider(_make_settings()) as provider:
            result = provider.invoke("test")
        assert result == "ok"
