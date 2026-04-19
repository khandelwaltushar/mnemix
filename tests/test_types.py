"""Tests for mnemix.types: immutability and validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mnemix.types import CacheConfig, CacheEntry, CacheResult, MetricsSnapshot


def _entry(**overrides: object) -> CacheEntry:
    defaults: dict[str, object] = {
        "id": "q1",
        "namespace": "default",
        "query_text": "hello world",
        "query_embedding": [0.1, 0.2, 0.3],
        "response": {"choices": [{"text": "hi"}]},
    }
    defaults.update(overrides)
    return CacheEntry(**defaults)  # type: ignore[arg-type]


class TestCacheEntry:
    def test_defaults(self) -> None:
        e = _entry()
        assert e.hit_count == 0
        assert e.ttl_seconds is None
        assert e.bypass is False
        assert e.created_at.tzinfo is UTC

    def test_is_frozen(self) -> None:
        e = _entry()
        with pytest.raises(ValidationError):
            e.hit_count = 5  # type: ignore[misc]

    def test_model_copy_produces_new_instance(self) -> None:
        e = _entry()
        bumped = e.model_copy(update={"hit_count": 3})
        assert bumped.hit_count == 3
        assert e.hit_count == 0

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _entry(unexpected="nope")

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            _entry(id="")

    def test_rejects_empty_namespace(self) -> None:
        with pytest.raises(ValidationError):
            _entry(namespace="")

    def test_rejects_empty_query_text(self) -> None:
        with pytest.raises(ValidationError):
            _entry(query_text="")

    def test_rejects_empty_embedding(self) -> None:
        with pytest.raises(ValidationError):
            _entry(query_embedding=[])

    def test_rejects_negative_hit_count(self) -> None:
        with pytest.raises(ValidationError):
            _entry(hit_count=-1)

    def test_rejects_nonpositive_ttl(self) -> None:
        with pytest.raises(ValidationError):
            _entry(ttl_seconds=0)

    def test_custom_created_at_preserved(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        assert _entry(created_at=ts).created_at == ts


class TestCacheResult:
    def test_miss(self) -> None:
        r = CacheResult(hit=False, latency_ms=2.5)
        assert r.entry is None
        assert r.similarity_score is None

    def test_hit_requires_entry(self) -> None:
        with pytest.raises(ValidationError):
            CacheResult(hit=True, entry=None, similarity_score=0.95, latency_ms=1.0)

    def test_hit_requires_similarity(self) -> None:
        with pytest.raises(ValidationError):
            CacheResult(hit=True, entry=_entry(), similarity_score=None, latency_ms=1.0)

    def test_miss_cannot_carry_entry(self) -> None:
        with pytest.raises(ValidationError):
            CacheResult(hit=False, entry=_entry(), similarity_score=None, latency_ms=1.0)

    def test_valid_hit(self) -> None:
        r = CacheResult(hit=True, entry=_entry(), similarity_score=0.99, latency_ms=0.5)
        assert r.entry is not None
        assert r.similarity_score == 0.99

    def test_is_frozen(self) -> None:
        r = CacheResult(hit=False, latency_ms=1.0)
        with pytest.raises(ValidationError):
            r.hit = True  # type: ignore[misc]

    def test_similarity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CacheResult(hit=False, similarity_score=1.5, latency_ms=1.0)

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CacheResult(hit=False, latency_ms=-0.1)


class TestCacheConfig:
    def test_defaults(self) -> None:
        c = CacheConfig()
        assert c.similarity_threshold == 0.92
        assert c.max_cache_size == 10_000
        assert c.default_ttl is None
        assert c.embedding_model == "all-MiniLM-L6-v2"

    def test_is_frozen(self) -> None:
        c = CacheConfig()
        with pytest.raises(ValidationError):
            c.similarity_threshold = 0.5  # type: ignore[misc]

    def test_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(similarity_threshold=1.1)
        with pytest.raises(ValidationError):
            CacheConfig(similarity_threshold=-0.01)

    def test_max_cache_size_positive(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(max_cache_size=0)

    def test_default_ttl_positive(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(default_ttl=0)

    def test_embedding_model_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(embedding_model="")


class TestMetricsSnapshot:
    def _snap(self, **overrides: object) -> MetricsSnapshot:
        defaults: dict[str, object] = {
            "total_requests": 10,
            "hits": 4,
            "misses": 6,
            "hit_rate": 0.4,
            "avg_similarity_on_hit": 0.95,
            "estimated_tokens_saved": 1200,
            "estimated_cost_saved_usd": 0.018,
        }
        defaults.update(overrides)
        return MetricsSnapshot(**defaults)  # type: ignore[arg-type]

    def test_valid(self) -> None:
        s = self._snap()
        assert s.hit_rate == 0.4

    def test_is_frozen(self) -> None:
        s = self._snap()
        with pytest.raises(ValidationError):
            s.hits = 99  # type: ignore[misc]

    def test_counters_must_add_up(self) -> None:
        with pytest.raises(ValidationError):
            self._snap(hits=5, misses=6, total_requests=10)

    def test_rejects_negative_counters(self) -> None:
        with pytest.raises(ValidationError):
            self._snap(hits=-1, misses=11, total_requests=10)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            self._snap(estimated_cost_saved_usd=-0.01)

    def test_hit_rate_clamped(self) -> None:
        with pytest.raises(ValidationError):
            self._snap(hit_rate=1.5)

    def test_zero_state(self) -> None:
        s = self._snap(
            total_requests=0,
            hits=0,
            misses=0,
            hit_rate=0.0,
            avg_similarity_on_hit=0.0,
            estimated_tokens_saved=0,
            estimated_cost_saved_usd=0.0,
        )
        assert s.total_requests == 0
