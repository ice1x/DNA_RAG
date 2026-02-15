"""Thread-safe in-memory LRU cache with TTL support.

Implements the :class:`~dna_rag.cache.base.Cache` protocol backed by
:class:`collections.OrderedDict` for O(1) LRU eviction.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class InMemoryCache:
    """In-memory LRU cache with per-entry TTL.

    Args:
        max_size: Maximum number of entries **per namespace**.
            When exceeded the least-recently-used entry is evicted.
        ttl_seconds: Time-to-live in seconds.  Entries older than this
            are treated as cache misses.  Set to ``0`` to disable TTL.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._stores: dict[str, OrderedDict[str, tuple[float, Any]]] = {}
        self._lock = Lock()

    # --- Cache protocol implementation ------------------------------------

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value or ``None`` on miss / expiry."""
        with self._lock:
            store = self._stores.get(namespace)
            if store is None:
                return None
            entry = store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if self._ttl > 0 and (time.monotonic() - ts) > self._ttl:
                del store[key]
                return None
            store.move_to_end(key)
            return value

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store a value, evicting the LRU entry if the namespace is full."""
        with self._lock:
            store = self._stores.setdefault(namespace, OrderedDict())
            store[key] = (time.monotonic(), value)
            store.move_to_end(key)
            while len(store) > self._max_size > 0:
                store.popitem(last=False)

    def invalidate(self, namespace: str, key: str) -> None:
        """Remove a specific entry (no-op if missing)."""
        with self._lock:
            store = self._stores.get(namespace)
            if store is not None:
                store.pop(key, None)

    def clear(self, namespace: str | None = None) -> None:
        """Clear all entries, or only those in *namespace*."""
        with self._lock:
            if namespace is None:
                self._stores.clear()
            else:
                self._stores.pop(namespace, None)
