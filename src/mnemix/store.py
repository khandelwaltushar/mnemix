"""Cache store and similarity search for Mnemix.

Defines the :class:`CacheStore` protocol, an :class:`InMemoryStore`
implementation with LRU eviction, a :func:`cosine_similarity` helper, and a
:class:`SimilarityIndex` that finds the best-matching :class:`CacheEntry`
for a query embedding within a namespace.

Example:
    Populate a store and search it::

        >>> import asyncio
        >>> from mnemix.store import InMemoryStore, SimilarityIndex
        >>> from mnemix.types import CacheEntry
        >>> async def demo() -> bool:
        ...     store = InMemoryStore(max_size=100)
        ...     entry = CacheEntry(
        ...         id="q1", namespace="user_a", query_text="hello",
        ...         query_embedding=[1.0, 0.0, 0.0], response={"text": "hi"},
        ...     )
        ...     await store.set(entry)
        ...     index = SimilarityIndex(store, threshold=0.9)
        ...     result = await index.search([1.0, 0.0, 0.0], namespace="user_a")
        ...     return result.hit
        >>> asyncio.run(demo())
        True
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mnemix.types import CacheEntry, CacheResult


class StoreStats(BaseModel):
    """Summary of the entries a :class:`CacheStore` holds right now.

    Attributes:
        total_entries: Total number of entries across all namespaces.
        entries_per_namespace: Mapping from namespace to entry count.

    Example:
        >>> StoreStats(total_entries=3, entries_per_namespace={"a": 2, "b": 1}).total_entries
        3
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_entries: int = Field(..., ge=0)
    entries_per_namespace: dict[str, int] = Field(default_factory=dict)


@runtime_checkable
class CacheStore(Protocol):
    """Protocol for persistent or in-memory cache backends.

    Implementations are namespace-scoped: the same ``key`` in different
    namespaces identifies different entries. All methods are coroutines so
    network-backed stores (Redis, etc.) can be dropped in without signature
    changes.
    """

    async def get(self, namespace: str, key: str) -> CacheEntry | None:
        """Retrieve an entry by (namespace, key), or return ``None`` if absent."""
        ...

    async def set(self, entry: CacheEntry) -> None:
        """Insert or overwrite an entry (namespace and key come from the entry)."""
        ...

    async def delete(self, namespace: str, key: str) -> bool:
        """Remove an entry. Returns True if it existed, False otherwise."""
        ...

    async def list_keys(self, namespace: str) -> list[str]:
        """Return all keys present in ``namespace``."""
        ...

    async def clear(self, namespace: str | None = None) -> int:
        """Remove all entries (or all entries in one namespace). Returns the count deleted."""
        ...

    async def stats(self) -> StoreStats:
        """Return a point-in-time :class:`StoreStats`."""
        ...


class InMemoryStore:
    """Process-local cache store with LRU eviction.

    Entries are held in an :class:`collections.OrderedDict` keyed by
    ``"{namespace}:{id}"``. Insertion order tracks recency: ``set`` moves the
    key to the end (most-recently-written), and when the store is at
    ``max_size`` a new insertion evicts the oldest key.

    ``get`` does *not* touch recency — it is a pure read. To mark an entry
    as recently used (e.g. after a cache hit), call ``set`` again with the
    updated entry. This keeps LRU semantics predictable under read-heavy
    similarity-search workloads that peek every entry in a namespace.

    TTL is not enforced by the store; :class:`SimilarityIndex` filters
    expired entries during search so the store stays a dumb key-value.

    Example:
        >>> import asyncio
        >>> from mnemix.types import CacheEntry
        >>> async def demo() -> int:
        ...     s = InMemoryStore(max_size=2)
        ...     for i in range(3):
        ...         await s.set(CacheEntry(
        ...             id=f"q{i}", namespace="ns", query_text=str(i),
        ...             query_embedding=[float(i)], response={},
        ...         ))
        ...     return len(await s.list_keys("ns"))
        >>> asyncio.run(demo())
        2
    """

    def __init__(self, max_size: int = 10_000) -> None:
        """Create an empty store.

        Args:
            max_size: Maximum number of entries retained across all
                namespaces. Must be positive.

        Raises:
            ValueError: If ``max_size`` is not positive.
        """
        if max_size <= 0:
            msg = f"max_size must be positive, got {max_size}"
            raise ValueError(msg)
        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    @staticmethod
    def _composite(namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    async def get(self, namespace: str, key: str) -> CacheEntry | None:
        """Peek an entry without affecting LRU order."""
        return self._data.get(self._composite(namespace, key))

    async def set(self, entry: CacheEntry) -> None:
        """Insert or update an entry; evicts the LRU key when at capacity."""
        composite = self._composite(entry.namespace, entry.id)
        if composite in self._data:
            self._data[composite] = entry
            self._data.move_to_end(composite)
            return
        if len(self._data) >= self._max_size:
            self._data.popitem(last=False)
        self._data[composite] = entry

    async def delete(self, namespace: str, key: str) -> bool:
        """Remove the entry if present. Returns whether a deletion happened."""
        composite = self._composite(namespace, key)
        if composite in self._data:
            del self._data[composite]
            return True
        return False

    async def list_keys(self, namespace: str) -> list[str]:
        """Return the entry IDs present in ``namespace`` (unordered)."""
        prefix = f"{namespace}:"
        return [k[len(prefix) :] for k in self._data if k.startswith(prefix)]

    async def clear(self, namespace: str | None = None) -> int:
        """Drop entries and return the count removed.

        Args:
            namespace: If provided, remove only that namespace; otherwise
                remove every entry.
        """
        if namespace is None:
            count = len(self._data)
            self._data.clear()
            return count
        prefix = f"{namespace}:"
        to_delete = [k for k in self._data if k.startswith(prefix)]
        for k in to_delete:
            del self._data[k]
        return len(to_delete)

    async def stats(self) -> StoreStats:
        """Return current entry counts, overall and per namespace."""
        per_ns: dict[str, int] = {}
        for composite in self._data:
            ns = composite.split(":", 1)[0]
            per_ns[ns] = per_ns.get(ns, 0) + 1
        return StoreStats(total_entries=len(self._data), entries_per_namespace=per_ns)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector. Must have the same length as ``a``.

    Returns:
        The cosine similarity in ``[-1.0, 1.0]``. Returns ``0.0`` if either
        vector is all zeros (rather than raising on division by zero).

    Raises:
        ValueError: If the vectors differ in length.

    Example:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
    """
    if len(a) != len(b):
        msg = f"vector length mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    score = float(np.dot(va, vb) / (na * nb))
    # Clamp float32 drift: identical float32 vectors can yield 1.0 + ε.
    return max(-1.0, min(1.0, score))


class SimilarityIndex:
    """Brute-force cosine similarity search over a :class:`CacheStore`.

    Iterates every entry in the requested namespace and returns the
    highest-scoring entry whose similarity to the query is ``>= threshold``.
    Entries with ``bypass=True`` and entries past their TTL are skipped.

    For the in-memory store this is O(N) per query but N is bounded by
    ``max_cache_size``. A future Redis-backed index can swap in native
    vector search without changing callers.

    Example:
        See the module docstring for a runnable example.
    """

    def __init__(self, store: CacheStore, threshold: float = 0.92) -> None:
        """Create an index over ``store``.

        Args:
            store: Any :class:`CacheStore` implementation.
            threshold: Minimum cosine similarity to treat a candidate as a
                hit. Must be in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``threshold`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be in [0.0, 1.0], got {threshold}"
            raise ValueError(msg)
        self._store = store
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Minimum cosine similarity required for a hit."""
        return self._threshold

    async def search(self, query_embedding: list[float], namespace: str) -> CacheResult:
        """Find the best matching entry in ``namespace`` for ``query_embedding``.

        Args:
            query_embedding: The query's dense vector.
            namespace: Namespace to restrict the search to.

        Returns:
            A :class:`CacheResult` — ``hit=True`` when an entry scored at or
            above the threshold; otherwise ``hit=False`` with the elapsed
            ``latency_ms``.
        """
        start = time.perf_counter()
        keys = await self._store.list_keys(namespace)
        best_entry: CacheEntry | None = None
        best_score = -1.0
        now = datetime.now(tz=UTC)
        for key in keys:
            entry = await self._store.get(namespace, key)
            if entry is None or entry.bypass or _is_expired(entry, now=now):
                continue
            score = cosine_similarity(query_embedding, entry.query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
        latency_ms = (time.perf_counter() - start) * 1000.0
        if best_entry is not None and best_score >= self._threshold:
            return CacheResult(
                hit=True,
                entry=best_entry,
                similarity_score=best_score,
                latency_ms=latency_ms,
            )
        return CacheResult(hit=False, latency_ms=latency_ms)


def _is_expired(entry: CacheEntry, *, now: datetime) -> bool:
    if entry.ttl_seconds is None:
        return False
    age = (now - entry.created_at).total_seconds()
    return age > entry.ttl_seconds
