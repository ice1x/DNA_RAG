"""DeepSeek LLM provider using direct HTTP calls via *httpx*.

Handles the ``<think>...</think>`` reasoning tokens that DeepSeek R1 models
emit, stripping them from the final response.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dna_rag.config import Settings
from dna_rag.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from dna_rag.logging import get_logger

logger = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class DeepSeekProvider:
    """LLM provider for DeepSeek's OpenAI-compatible API.

    Implements the :class:`~dna_rag.llm.base.LLMProvider` protocol.

    Args:
        settings: Application settings (API key, model name, etc.).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.llm_base_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=settings.llm_timeout,
        )
        # Build the tenacity retry wrapper with runtime settings.
        self._invoke_with_retry = retry(
            retry=retry_if_exception_type((LLMConnectionError, LLMRateLimitError)),
            stop=stop_after_attempt(max(1, settings.llm_max_retries)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )(self._invoke_once)

    # --- LLMProvider protocol ----------------------------------------------

    def invoke(self, prompt: str) -> str:
        """Send *prompt* to DeepSeek and return the cleaned response text."""
        return self._invoke_with_retry(prompt)

    # --- internals ---------------------------------------------------------

    def _invoke_once(self, prompt: str) -> str:
        """Single (non-retried) invocation."""
        logger.info(
            "llm_invoke",
            model=self._settings.llm_model,
            prompt_len=len(prompt),
        )

        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._settings.llm_temperature,
        }
        if self._settings.llm_max_tokens is not None:
            payload["max_tokens"] = self._settings.llm_max_tokens

        try:
            response = self._client.post("/chat/completions", json=payload)
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Cannot connect to {self._settings.llm_base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"Request timed out after {self._settings.llm_timeout}s"
            ) from exc

        if response.status_code == 429:
            raise LLMRateLimitError("Rate limit exceeded")
        if response.status_code >= 400:
            raise LLMResponseError(
                f"API error {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMResponseError(
                f"Unexpected API response structure: {response.text[:500]}"
            ) from exc

        # Strip <think>...</think> reasoning blocks emitted by R1 models.
        content = _THINK_RE.sub("", content).strip()

        logger.info("llm_response", response_len=len(content))
        return content

    # --- resource management -----------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> DeepSeekProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
