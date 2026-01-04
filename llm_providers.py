"""Abstract LLM provider layer supporting multiple providers with priority fallback.

This module provides a unified interface for different LLM providers (OpenAI, DeepSeek, etc.)
with automatic fallback based on configured priorities.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMMessage(BaseModel):
    """Standard message format for LLM communication."""

    role: str = Field(description="Message role: system, user, or assistant")
    content: str = Field(description="Message content")


class LLMResponse(BaseModel):
    """Standard response format from LLM providers."""

    content: str = Field(description="Generated text response")
    provider: str = Field(description="Provider that generated the response")
    model: str = Field(description="Model used for generation")
    usage: dict[str, int] | None = Field(default=None, description="Token usage statistics")


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        """Initialize LLM provider.

        Parameters
        ----------
        api_key : str
            API key for authentication
        model : str
            Model identifier
        temperature : float
            Sampling temperature (0.0-2.0)
        max_retries : int
            Maximum number of retry attempts
        timeout : float
            Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()

    @abstractmethod
    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate completion from messages.

        Parameters
        ----------
        messages : list[LLMMessage]
            List of conversation messages
        **kwargs : Any
            Additional provider-specific parameters

        Returns
        -------
        LLMResponse
            Generated response
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name.

        Returns
        -------
        str
            Provider identifier
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is available (has valid API key).

        Returns
        -------
        bool
            True if provider can be used
        """
        return bool(self.api_key)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 30.0,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        """Initialize OpenAI provider.

        Parameters
        ----------
        api_key : str
            OpenAI API key
        model : str
            Model name (default: gpt-4o-mini)
        temperature : float
            Sampling temperature
        max_retries : int
            Maximum retry attempts
        timeout : float
            Request timeout
        base_url : str
            API base URL (for compatible services)
        """
        super().__init__(api_key, model, temperature, max_retries, timeout)
        self.base_url = base_url

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate completion using OpenAI API.

        Parameters
        ----------
        messages : list[LLMMessage]
            Conversation messages
        **kwargs : Any
            Additional parameters (max_tokens, etc.)

        Returns
        -------
        LLMResponse
            Generated response
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add optional parameters
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider=self.get_provider_name(),
                    model=self.model,
                    usage=data.get("usage"),
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenAI API request failed after {self.max_retries} retries: {exc}"
                    ) from exc
                logger.warning(f"OpenAI request attempt {attempt + 1} failed: {exc}")

        raise RuntimeError("Unexpected error in OpenAI provider")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 30.0,
        base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        """Initialize DeepSeek provider.

        Parameters
        ----------
        api_key : str
            DeepSeek API key
        model : str
            Model name (default: deepseek-chat)
        temperature : float
            Sampling temperature
        max_retries : int
            Maximum retry attempts
        timeout : float
            Request timeout
        base_url : str
            API base URL
        """
        super().__init__(api_key, model, temperature, max_retries, timeout)
        self.base_url = base_url

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate completion using DeepSeek API.

        Parameters
        ----------
        messages : list[LLMMessage]
            Conversation messages
        **kwargs : Any
            Additional parameters

        Returns
        -------
        LLMResponse
            Generated response
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    provider=self.get_provider_name(),
                    model=self.model,
                    usage=data.get("usage"),
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"DeepSeek API request failed after {self.max_retries} retries: {exc}"
                    ) from exc
                logger.warning(f"DeepSeek request attempt {attempt + 1} failed: {exc}")

        raise RuntimeError("Unexpected error in DeepSeek provider")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "deepseek"


class LLMProviderManager:
    """Manager for multiple LLM providers with priority-based fallback."""

    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        """Initialize provider manager.

        Parameters
        ----------
        providers : list[BaseLLMProvider]
            List of providers in priority order (first = highest priority)
        """
        self.providers = [p for p in providers if p.is_available()]
        if not self.providers:
            raise ValueError("No available LLM providers configured")

        logger.info(
            f"Initialized LLM manager with {len(self.providers)} provider(s): "
            f"{[p.get_provider_name() for p in self.providers]}"
        )

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate completion using first available provider.

        Tries providers in priority order until one succeeds.

        Parameters
        ----------
        messages : list[LLMMessage]
            Conversation messages
        **kwargs : Any
            Additional parameters

        Returns
        -------
        LLMResponse
            Generated response

        Raises
        ------
        RuntimeError
            If all providers fail
        """
        last_error = None

        for provider in self.providers:
            try:
                logger.debug(f"Attempting generation with {provider.get_provider_name()}")
                response = provider.generate(messages, **kwargs)
                logger.info(f"Successfully generated response using {response.provider}")
                return response
            except Exception as exc:
                logger.warning(f"Provider {provider.get_provider_name()} failed: {exc}")
                last_error = exc
                continue

        raise RuntimeError(
            f"All {len(self.providers)} LLM provider(s) failed. Last error: {last_error}"
        ) from last_error

    def get_active_provider(self) -> str:
        """Get name of first available provider.

        Returns
        -------
        str
            Provider name
        """
        return self.providers[0].get_provider_name() if self.providers else "none"
