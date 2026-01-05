"""Tests for LLM provider abstraction layer."""

from unittest.mock import Mock, patch

import pytest
import requests

from dna_rag.core.llm_providers import (
    DeepSeekProvider,
    LLMMessage,
    LLMProviderManager,
    LLMResponse,
    OpenAIProvider,
)


def test_llm_message_creation():
    """Test creating LLM messages."""
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_llm_response_creation():
    """Test creating LLM responses."""
    response = LLMResponse(
        content="Hi there",
        provider="openai",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    assert response.content == "Hi there"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.usage["prompt_tokens"] == 10


@patch("llm_providers.requests.Session.post")
def test_openai_provider_generate(mock_post):
    """Test OpenAI provider generation."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenAI response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="test_key", model="gpt-4o-mini")
    messages = [LLMMessage(role="user", content="Test")]

    response = provider.generate(messages)

    assert response.content == "OpenAI response"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert mock_post.called


@patch("llm_providers.requests.Session.post")
def test_deepseek_provider_generate(mock_post):
    """Test DeepSeek provider generation."""
    # Mock successful response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "DeepSeek response"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    provider = DeepSeekProvider(api_key="test_key", model="deepseek-chat")
    messages = [LLMMessage(role="user", content="Test")]

    response = provider.generate(messages)

    assert response.content == "DeepSeek response"
    assert response.provider == "deepseek"
    assert response.model == "deepseek-chat"
    assert mock_post.called


@patch("llm_providers.requests.Session.post")
def test_openai_provider_retry(mock_post):
    """Test OpenAI provider retry on failure."""
    # First call fails, second succeeds
    mock_response_fail = Mock()
    mock_response_fail.raise_for_status.side_effect = requests.RequestException("Error")

    mock_response_success = Mock()
    mock_response_success.json.return_value = {
        "choices": [{"message": {"content": "Success"}}],
        "usage": {},
    }
    mock_response_success.raise_for_status.return_value = None

    mock_post.side_effect = [mock_response_fail, mock_response_success]

    provider = OpenAIProvider(api_key="test_key", max_retries=2)
    messages = [LLMMessage(role="user", content="Test")]

    response = provider.generate(messages)
    assert response.content == "Success"
    assert mock_post.call_count == 2


@patch("llm_providers.requests.Session.post")
def test_openai_provider_max_retries_exceeded(mock_post):
    """Test OpenAI provider fails after max retries."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.RequestException("Error")
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="test_key", max_retries=1)
    messages = [LLMMessage(role="user", content="Test")]

    with pytest.raises(RuntimeError, match="failed after 1 retries"):
        provider.generate(messages)

    assert mock_post.call_count == 2  # Initial + 1 retry


def test_provider_is_available():
    """Test provider availability check."""
    provider_with_key = OpenAIProvider(api_key="test_key")
    assert provider_with_key.is_available()

    provider_without_key = OpenAIProvider(api_key="")
    assert not provider_without_key.is_available()


def test_llm_manager_initialization():
    """Test LLM manager initialization with multiple providers."""
    providers = [
        OpenAIProvider(api_key="openai_key"),
        DeepSeekProvider(api_key="deepseek_key"),
    ]

    manager = LLMProviderManager(providers)
    assert len(manager.providers) == 2
    assert manager.get_active_provider() == "openai"


def test_llm_manager_filters_unavailable():
    """Test LLM manager filters out providers without API keys."""
    providers = [
        OpenAIProvider(api_key=""),  # No key
        DeepSeekProvider(api_key="deepseek_key"),
    ]

    manager = LLMProviderManager(providers)
    assert len(manager.providers) == 1
    assert manager.get_active_provider() == "deepseek"


def test_llm_manager_no_providers():
    """Test LLM manager raises error with no available providers."""
    providers = [
        OpenAIProvider(api_key=""),
        DeepSeekProvider(api_key=""),
    ]

    with pytest.raises(ValueError, match="No available LLM providers"):
        LLMProviderManager(providers)


@patch("llm_providers.requests.Session.post")
def test_llm_manager_fallback(mock_post):
    """Test LLM manager fallback to second provider on failure."""
    # First provider fails, second succeeds
    mock_response_fail = Mock()
    mock_response_fail.raise_for_status.side_effect = requests.RequestException("Error")

    mock_response_success = Mock()
    mock_response_success.json.return_value = {
        "choices": [{"message": {"content": "DeepSeek fallback"}}],
        "usage": {},
    }
    mock_response_success.raise_for_status.return_value = None

    # OpenAI fails (2 attempts), DeepSeek succeeds
    mock_post.side_effect = [
        mock_response_fail,
        mock_response_fail,
        mock_response_success,
    ]

    providers = [
        OpenAIProvider(api_key="openai_key", max_retries=1),
        DeepSeekProvider(api_key="deepseek_key"),
    ]

    manager = LLMProviderManager(providers)
    messages = [LLMMessage(role="user", content="Test")]

    response = manager.generate(messages)
    assert response.content == "DeepSeek fallback"
    assert response.provider == "deepseek"


@patch("llm_providers.requests.Session.post")
def test_llm_manager_all_providers_fail(mock_post):
    """Test LLM manager raises error when all providers fail."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.RequestException("Error")
    mock_post.return_value = mock_response

    providers = [
        OpenAIProvider(api_key="openai_key", max_retries=0),
        DeepSeekProvider(api_key="deepseek_key", max_retries=0),
    ]

    manager = LLMProviderManager(providers)
    messages = [LLMMessage(role="user", content="Test")]

    with pytest.raises(RuntimeError, match="All 2 LLM provider\\(s\\) failed"):
        manager.generate(messages)


def test_openai_provider_custom_parameters():
    """Test OpenAI provider with custom parameters."""
    provider = OpenAIProvider(
        api_key="test_key",
        model="gpt-4",
        temperature=0.7,
        max_retries=3,
        timeout=60.0,
        base_url="https://custom.api.com/v1",
    )

    assert provider.model == "gpt-4"
    assert provider.temperature == 0.7
    assert provider.max_retries == 3
    assert provider.timeout == 60.0
    assert provider.base_url == "https://custom.api.com/v1"


def test_deepseek_provider_custom_parameters():
    """Test DeepSeek provider with custom parameters."""
    provider = DeepSeekProvider(
        api_key="test_key",
        model="deepseek-coder",
        temperature=0.5,
        max_retries=5,
        timeout=45.0,
        base_url="https://custom.deepseek.com/v1",
    )

    assert provider.model == "deepseek-coder"
    assert provider.temperature == 0.5
    assert provider.max_retries == 5
    assert provider.timeout == 45.0
    assert provider.base_url == "https://custom.deepseek.com/v1"
