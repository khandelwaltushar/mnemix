"""Tests for mnemix.store: InMemoryStore, cosine_similarity, SimilarityIndex."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mnemix.store import (
    CacheStore,
    InMemoryStore,
    SimilarityIndex,
    StoreStats,
    cosine_similarity,
)
from mnemix.types import CacheEntry


def _entry(
    *,
    id: str = "q1",
    namespace: str = "default",
    embedding: list[float] | None = None,
    bypass: bool = False,
    ttl_seconds: int | None = None,
    created_at: datetime | None = None,
) -> CacheEntry:
    return CacheEntry(
        id=id,
        namespace=namespace,
        query_text=f"text for {id}",
        query_embedding=embedding if embedding is not None else [1.0, 0.0, 0.0],
        response={"text": f"resp {id}"},
        bypass=bypass,
        ttl_seconds=ttl_seconds,
        created_at=created_at or datetime.now(tz=UTC),
    )


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestInMemoryStore:
    async def test_conforms_to_protocol(self) -> None:
        store = InMemoryStore()
        assert isinstance(store, CacheStore)

    async def test_rejects_nonpositive_max_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InMemoryStore(max_size=0)
        with pytest.raises(ValueError, match="positive"):
            InMemoryStore(max_size=-1)

    async def test_set_then_get(self) -> None:
        store = InMemoryStore()
        entry = _entry(id="q1", namespace="ns")
        await store.set(entry)
        got = await store.get("ns", "q1")
        assert got == entry

    async def test_get_missing_returns_none(self) -> None:
        store = InMemoryStore()
        assert await store.get("ns", "nope") is None

    async def test_delete(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="ns"))
        assert await store.delete("ns", "q1") is True
        assert await store.delete("ns", "q1") is False
        assert await store.get("ns", "q1") is None

    async def test_list_keys_namespace_filtered(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="a"))
        await store.set(_entry(id="q3", namespace="b"))
        keys_a = sorted(await store.list_keys("a"))
        keys_b = await store.list_keys("b")
        assert keys_a == ["q1", "q2"]
        assert keys_b == ["q3"]

    async def test_clear_all(self) -> None:
        store = InMemoryStore()
        for i in range(3):
            await store.set(_entry(id=f"q{i}", namespace="ns"))
        removed = await store.clear()
        assert removed == 3
        assert await store.list_keys("ns") == []

    async def test_clear_one_namespace(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="b"))
        removed = await store.clear("a")
        assert removed == 1
        assert await store.list_keys("a") == []
        assert await store.list_keys("b") == ["q2"]

    async def test_stats(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="a"))
        await store.set(_entry(id="q3", namespace="b"))
        stats = await store.stats()
        assert isinstance(stats, StoreStats)
        assert stats.total_entries == 3
        assert stats.entries_per_namespace == {"a": 2, "b": 1}

    async def test_lru_eviction_on_capacity(self) -> None:
        store = InMemoryStore(max_size=2)
        await store.set(_entry(id="q1", namespace="ns"))
        await store.set(_entry(id="q2", namespace="ns"))
        await store.set(_entry(id="q3", namespace="ns"))  # evicts q1

        assert await store.get("ns", "q1") is None
        assert await store.get("ns", "q2") is not None
        assert await store.get("ns", "q3") is not None

    async def test_set_refresh_moves_to_end(self) -> None:
        store = InMemoryStore(max_size=2)
        await store.set(_entry(id="q1", namespace="ns"))
        await store.set(_entry(id="q2", namespace="ns"))
        # refresh q1 so q2 becomes oldest
        await store.set(_entry(id="q1", namespace="ns", embedding=[0.5, 0.5, 0.0]))
        await store.set(_entry(id="q3", namespace="ns"))  # evicts q2

        assert await store.get("ns", "q1") is not None
        assert await store.get("ns", "q2") is None
        assert await store.get("ns", "q3") is not None

    async def test_get_does_not_bump_lru(self) -> None:
        # Critical: reads must be side-effect-free so similarity scans don't
        # perturb eviction order.
        store = InMemoryStore(max_size=2)
        await store.set(_entry(id="q1", namespace="ns"))
        await store.set(_entry(id="q2", namespace="ns"))
        await store.get("ns", "q1")  # reading q1 should NOT bump it
        await store.set(_entry(id="q3", namespace="ns"))  # still evicts q1

        assert await store.get("ns", "q1") is None
        assert await store.get("ns", "q2") is not None
        assert await store.get("ns", "q3") is not None

    async def test_namespace_isolation_on_get(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="shared_id", namespace="user_a"))
        assert await store.get("user_b", "shared_id") is None


class TestSimilarityIndex:
    async def test_rejects_bad_threshold(self) -> None:
        store = InMemoryStore()
        with pytest.raises(ValueError, match="threshold"):
            SimilarityIndex(store, threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            SimilarityIndex(store, threshold=1.01)

    async def test_identical_query_always_hits(self) -> None:
        store = InMemoryStore()
        vec = [0.1, 0.2, 0.3, 0.4]
        await store.set(_entry(id="q1", namespace="ns", embedding=vec))
        index = SimilarityIndex(store, threshold=0.92)
        result = await index.search(vec, namespace="ns")
        assert result.hit is True
        assert result.entry is not None
        assert result.entry.id == "q1"
        assert result.similarity_score == pytest.approx(1.0)

    async def test_threshold_respected_above(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="ns", embedding=[1.0, 0.0, 0.0]))
        index = SimilarityIndex(store, threshold=0.95)
        # Nearly parallel — similarity ≈ 0.9999
        result = await index.search([1.0, 0.01, 0.0], namespace="ns")
        assert result.hit is True

    async def test_threshold_respected_below(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="ns", embedding=[1.0, 0.0, 0.0]))
        index = SimilarityIndex(store, threshold=0.95)
        # Orthogonal — similarity = 0
        result = await index.search([0.0, 1.0, 0.0], namespace="ns")
        assert result.hit is False
        assert result.entry is None

    async def test_empty_namespace_misses(self) -> None:
        store = InMemoryStore()
        index = SimilarityIndex(store, threshold=0.5)
        result = await index.search([1.0, 0.0], namespace="ns")
        assert result.hit is False
        assert result.latency_ms >= 0.0

    async def test_returns_best_of_multiple(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="far", namespace="ns", embedding=[0.0, 1.0, 0.0]))
        await store.set(_entry(id="near", namespace="ns", embedding=[1.0, 0.05, 0.0]))
        await store.set(_entry(id="mid", namespace="ns", embedding=[0.7, 0.7, 0.0]))
        index = SimilarityIndex(store, threshold=0.5)
        result = await index.search([1.0, 0.0, 0.0], namespace="ns")
        assert result.hit is True
        assert result.entry is not None
        assert result.entry.id == "near"

    async def test_namespace_isolation(self) -> None:
        store = InMemoryStore()
        await store.set(_entry(id="q1", namespace="user_a", embedding=[1.0, 0.0]))
        index = SimilarityIndex(store, threshold=0.5)
        # Same exact vector in user_b — must not leak from user_a
        result = await index.search([1.0, 0.0], namespace="user_b")
        assert result.hit is False

    async def test_bypass_entries_are_skipped(self) -> None:
        store = InMemoryStore()
        await store.set(
            _entry(id="q1", namespace="ns", embedding=[1.0, 0.0], bypass=True),
        )
        index = SimilarityIndex(store, threshold=0.5)
        result = await index.search([1.0, 0.0], namespace="ns")
        assert result.hit is False

    async def test_expired_entries_are_skipped(self) -> None:
        store = InMemoryStore()
        old = datetime.now(tz=UTC) - timedelta(seconds=120)
        await store.set(
            _entry(
                id="q1",
                namespace="ns",
                embedding=[1.0, 0.0],
                ttl_seconds=60,
                created_at=old,
            ),
        )
        index = SimilarityIndex(store, threshold=0.5)
        result = await index.search([1.0, 0.0], namespace="ns")
        assert result.hit is False

    async def test_unexpired_ttl_entries_hit(self) -> None:
        store = InMemoryStore()
        recent = datetime.now(tz=UTC) - timedelta(seconds=5)
        await store.set(
            _entry(
                id="q1",
                namespace="ns",
                embedding=[1.0, 0.0],
                ttl_seconds=60,
                created_at=recent,
            ),
        )
        index = SimilarityIndex(store, threshold=0.5)
        result = await index.search([1.0, 0.0], namespace="ns")
        assert result.hit is True

    async def test_threshold_property_exposed(self) -> None:
        store = InMemoryStore()
        index = SimilarityIndex(store, threshold=0.77)
        assert index.threshold == 0.77
