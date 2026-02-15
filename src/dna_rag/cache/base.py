"""Cache protocol definition.

Any object that implements :class:`Cache` can be used as the caching backend
for :class:`~dna_rag.engine.DNAAnalysisEngine`.  The protocol uses
*namespaces* to logically separate different kinds of cached data (DNA files,
SNP lookups, answers) within a single backend instance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Structural sub-typing protocol for cache backends."""

    def get(self, namespace: str, key: str) -> Any | None:
        """Retrieve a cached value.  Returns ``None`` on cache miss."""
        ...

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store *value* under *key* in *namespace*."""
        ...

    def invalidate(self, namespace: str, key: str) -> None:
        """Remove a specific entry."""
        ...

    def clear(self, namespace: str | None = None) -> None:
        """Clear all entries, optionally scoped to *namespace*."""
        ...
