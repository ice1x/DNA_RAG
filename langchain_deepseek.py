"""Wrapper around the DeepSeek chat completion API.

This module provides a minimal interface compatible with the rest of the
project while delegating the heavy lifting to the real DeepSeek HTTP API.
The class mimics the interface of ``ChatOpenAI`` from LangChain: it is
callable with a list of ``langchain.schema`` messages and returns an object
with a ``content`` attribute containing the model response.

Network errors and timeouts are gracefully handled. In such cases an empty
response is returned so callers do not have to deal with exceptions.
"""

from __future__ import annotations

from typing import List, Optional

import requests
from langchain.schema import BaseMessage


class ChatDeepSeek:
    """Small wrapper around the DeepSeek chat completion API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key or ""
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"

    # ------------------------------------------------------------------
    def __call__(self, messages: List[BaseMessage]):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": getattr(m, "type", "user"),
                    "content": m.content,
                }
                for m in messages
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {"Authorization": f"Bearer {self.api_key}"}

        content = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                break
            except (requests.Timeout, requests.RequestException, KeyError, IndexError):
                if attempt == self.max_retries:
                    content = ""
                else:
                    continue

        class Result:
            def __init__(self, content: str) -> None:
                self.content = content

        return Result(content)

