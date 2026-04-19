"""Core data models for Mnemix.

All models are immutable pydantic v2 models (``frozen=True``). They define
the data that flows between the proxy, the cache store, and the metrics layer.

Example:
    Build a cache entry and assert it cannot be mutated::

        >>> from mnemix.types import CacheEntry
        >>> entry = CacheEntry(
        ...     id="abc",
        ...     namespace="default",
        ...     query_text="hello",
        ...     query_embedding=[0.1, 0.2],
        ...     response={"choices": []},
        ... )
        >>> entry.hit_count
        0
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CacheEntry(BaseModel):
    """A single cached LLM response keyed by a query embedding.

    Entries are immutable. To update a counter such as ``hit_count``, build a
    new entry with :meth:`pydantic.BaseModel.model_copy`.

    Attributes:
        id: Stable identifier (typically a content hash of the query).
        namespace: Logical grouping so tenants or models don't collide.
        query_text: Original request text used to produce the embedding.
        query_embedding: Dense vector representation of ``query_text``.
        response: Upstream provider response payload verbatim.
        hit_count: Number of times this entry has been served from cache.
        created_at: UTC timestamp at which the entry was first stored.
        ttl_seconds: Optional expiration in seconds from ``created_at``.
        bypass: When True, the entry was marked non-cacheable (e.g., a
            time-sensitive query) and should never be returned as a hit.

    Example:
        >>> entry = CacheEntry(
        ...     id="q1",
        ...     namespace="default",
        ...     query_text="what is 2+2?",
        ...     query_embedding=[0.0, 1.0],
        ...     response={"answer": "4"},
        ... )
        >>> entry.bypass
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1)
    namespace: str = Field(..., min_length=1)
    query_text: str = Field(..., min_length=1)
    query_embedding: list[float] = Field(..., min_length=1)
    response: dict[str, Any]
    hit_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    ttl_seconds: int | None = Field(default=None, gt=0)
    bypass: bool = False


class CacheResult(BaseModel):
    """Outcome of a cache lookup.

    Attributes:
        hit: True when a semantically similar entry was found and returned.
        entry: The matched entry on a hit, else ``None``.
        similarity_score: Cosine similarity on a hit, else ``None``.
        latency_ms: Wall-clock latency of the lookup in milliseconds.

    Example:
        >>> miss = CacheResult(hit=False, entry=None, similarity_score=None, latency_ms=1.2)
        >>> miss.hit
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hit: bool
    entry: CacheEntry | None = None
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _hit_requires_entry(self) -> Self:
        """Enforce that a hit carries an entry and a similarity score."""
        if self.hit and (self.entry is None or self.similarity_score is None):
            msg = "hit=True requires both 'entry' and 'similarity_score' to be set"
            raise ValueError(msg)
        if not self.hit and self.entry is not None:
            msg = "hit=False must not carry an 'entry'"
            raise ValueError(msg)
        return self


class CacheConfig(BaseModel):
    """Runtime configuration for the cache.

    Attributes:
        similarity_threshold: Minimum cosine similarity to treat a candidate
            as a hit. Values closer to 1.0 are stricter.
        max_cache_size: Upper bound on the number of entries retained in
            memory before eviction.
        default_ttl: Default time-to-live in seconds for new entries.
            ``None`` means entries never expire unless ``ttl_seconds`` is
            set explicitly on the entry.
        embedding_model: SentenceTransformer model name used by the local
            :class:`EmbeddingEngine` implementation.

    Example:
        >>> cfg = CacheConfig(similarity_threshold=0.95)
        >>> cfg.embedding_model
        'all-MiniLM-L6-v2'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    max_cache_size: int = Field(default=10_000, gt=0)
    default_ttl: int | None = Field(default=None, gt=0)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)


class MetricsSnapshot(BaseModel):
    """Point-in-time view of cache performance.

    Attributes:
        total_requests: Number of lookups observed.
        hits: Subset of ``total_requests`` that returned a cached response.
        misses: Subset of ``total_requests`` that were forwarded upstream.
        hit_rate: ``hits / total_requests``; ``0.0`` when there are no
            requests yet.
        avg_similarity_on_hit: Mean cosine similarity across hits; ``0.0``
            when there have been no hits.
        estimated_tokens_saved: Sum of upstream tokens avoided by serving
            from cache.
        estimated_cost_saved_usd: Dollar estimate derived from token counts
            and a per-model pricing table.

    Example:
        >>> snap = MetricsSnapshot(total_requests=10, hits=4, misses=6,
        ...                        hit_rate=0.4, avg_similarity_on_hit=0.95,
        ...                        estimated_tokens_saved=1200,
        ...                        estimated_cost_saved_usd=0.018)
        >>> snap.hit_rate
        0.4
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_requests: int = Field(..., ge=0)
    hits: int = Field(..., ge=0)
    misses: int = Field(..., ge=0)
    hit_rate: float = Field(..., ge=0.0, le=1.0)
    avg_similarity_on_hit: float = Field(..., ge=0.0, le=1.0)
    estimated_tokens_saved: int = Field(..., ge=0)
    estimated_cost_saved_usd: float = Field(..., ge=0.0)

    @field_validator("hit_rate")
    @classmethod
    def _round_hit_rate(cls, v: float) -> float:
        """Clamp floating-point drift to the documented [0.0, 1.0] range."""
        return max(0.0, min(1.0, v))

    @model_validator(mode="after")
    def _hits_plus_misses_equal_total(self) -> Self:
        """Ensure the counters are internally consistent."""
        if self.hits + self.misses != self.total_requests:
            msg = (
                f"hits ({self.hits}) + misses ({self.misses}) must equal "
                f"total_requests ({self.total_requests})"
            )
            raise ValueError(msg)
        return self
