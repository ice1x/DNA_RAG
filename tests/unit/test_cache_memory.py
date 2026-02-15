"""Tests for the in-memory LRU cache."""

from __future__ import annotations

import threading
import time

from dna_rag.cache.memory import InMemoryCache


class TestBasicOperations:
    """get / set / invalidate / clear."""

    def test_get_miss(self):
        cache = InMemoryCache()
        assert cache.get("ns", "key") is None

    def test_set_then_get(self):
        cache = InMemoryCache()
        cache.set("ns", "key", "value")
        assert cache.get("ns", "key") == "value"

    def test_overwrite(self):
        cache = InMemoryCache()
        cache.set("ns", "key", "v1")
        cache.set("ns", "key", "v2")
        assert cache.get("ns", "key") == "v2"

    def test_invalidate(self):
        cache = InMemoryCache()
        cache.set("ns", "key", "value")
        cache.invalidate("ns", "key")
        assert cache.get("ns", "key") is None

    def test_invalidate_missing_key(self):
        cache = InMemoryCache()
        cache.invalidate("ns", "missing")  # should not raise

    def test_clear_specific_namespace(self):
        cache = InMemoryCache()
        cache.set("ns1", "k", "v1")
        cache.set("ns2", "k", "v2")
        cache.clear("ns1")
        assert cache.get("ns1", "k") is None
        assert cache.get("ns2", "k") == "v2"

    def test_clear_all(self):
        cache = InMemoryCache()
        cache.set("ns1", "k", "v1")
        cache.set("ns2", "k", "v2")
        cache.clear()
        assert cache.get("ns1", "k") is None
        assert cache.get("ns2", "k") is None


class TestNamespaceIsolation:
    """Keys in different namespaces do not collide."""

    def test_same_key_different_namespaces(self):
        cache = InMemoryCache()
        cache.set("a", "key", "alpha")
        cache.set("b", "key", "beta")
        assert cache.get("a", "key") == "alpha"
        assert cache.get("b", "key") == "beta"


class TestTTL:
    """Time-to-live expiration."""

    def test_entry_expires_after_ttl(self):
        cache = InMemoryCache(ttl_seconds=1)
        cache.set("ns", "key", "value")
        assert cache.get("ns", "key") == "value"  # still valid
        time.sleep(1.1)  # wait past TTL
        assert cache.get("ns", "key") is None  # expired

    def test_entry_valid_within_ttl(self):
        cache = InMemoryCache(ttl_seconds=60)
        cache.set("ns", "key", "value")
        assert cache.get("ns", "key") == "value"


class TestLRUEviction:
    """Least-recently-used eviction when max_size is reached."""

    def test_evicts_oldest(self):
        cache = InMemoryCache(max_size=2, ttl_seconds=3600)
        cache.set("ns", "a", 1)
        cache.set("ns", "b", 2)
        cache.set("ns", "c", 3)  # 'a' should be evicted
        assert cache.get("ns", "a") is None
        assert cache.get("ns", "b") == 2
        assert cache.get("ns", "c") == 3

    def test_access_refreshes_order(self):
        cache = InMemoryCache(max_size=2, ttl_seconds=3600)
        cache.set("ns", "a", 1)
        cache.set("ns", "b", 2)
        cache.get("ns", "a")  # refresh 'a'
        cache.set("ns", "c", 3)  # 'b' should be evicted (oldest)
        assert cache.get("ns", "a") == 1
        assert cache.get("ns", "b") is None
        assert cache.get("ns", "c") == 3

    def test_max_size_zero_disables_eviction(self):
        cache = InMemoryCache(max_size=0, ttl_seconds=3600)
        cache.set("ns", "a", 1)
        cache.set("ns", "b", 2)
        # max_size=0 means unbounded (the while condition is False)
        assert cache.get("ns", "a") == 1
        assert cache.get("ns", "b") == 2


class TestThreadSafety:
    """Concurrent operations should not corrupt state."""

    def test_concurrent_writes(self):
        cache = InMemoryCache(max_size=1000, ttl_seconds=3600)
        errors: list[Exception] = []

        def writer(ns: str, count: int) -> None:
            try:
                for i in range(count):
                    cache.set(ns, f"key_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"ns_{t}", 100))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Spot check
        assert cache.get("ns_0", "key_0") == 0


class TestComplexValues:
    """Cache can store complex objects."""

    def test_stores_dataframe(self):
        import pandas as pd

        cache = InMemoryCache()
        df = pd.DataFrame({"RSID": ["rs1"], "GENOTYPE": ["AA"]})
        cache.set("ns", "df", df)
        retrieved = cache.get("ns", "df")
        assert retrieved is not None
        assert len(retrieved) == 1
