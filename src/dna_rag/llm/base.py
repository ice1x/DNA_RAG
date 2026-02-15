"""LLM provider protocol.

Any class implementing :class:`LLMProvider` can serve as the language model
backend for the DNA analysis engine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Structural sub-typing protocol for LLM providers.

    Implementations should raise the appropriate exception from
    :mod:`dna_rag.exceptions` on failure:

    * :class:`~dna_rag.exceptions.LLMConnectionError` -- network / timeout.
    * :class:`~dna_rag.exceptions.LLMResponseError` -- bad API response.
    * :class:`~dna_rag.exceptions.LLMRateLimitError` -- HTTP 429.
    """

    def invoke(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the text response."""
        ...
