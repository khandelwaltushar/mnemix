r"""Redis-backed cache store for Mnemix.

Implements :class:`~mnemix.store.CacheStore` against a Redis server using
the ``redis.asyncio`` client. Entries are serialised into Redis hashes and
indexed by namespace for enumeration. TTL is applied via native Redis
``EXPIRE``, so expired entries vanish automatically.

The implementation degrades gracefully when Redis is unreachable: all
operations catch :class:`redis.exceptions.RedisError`, log a warning, and
return a safe default (``None``, empty list, no-op). This keeps the proxy
responsive — it falls back to always-forward behaviour rather than
crashing when the cache tier is unavailable.

Storage layout (``key_prefix`` defaults to ``"mnemix"``):

* ``{prefix}:entry:{namespace}:{id}`` — hash holding a serialised
  :class:`~mnemix.types.CacheEntry`.
* ``{prefix}:ns:{namespace}`` — set of entry IDs in a namespace, used by
  :meth:`list_keys`.
* ``{prefix}:namespaces`` — set of all namespaces ever written, used by
  :meth:`stats` and full ``clear()``.

Production upgrade path — Redis VSS (Vector Similarity Search)
--------------------------------------------------------------
The default similarity search path in :class:`mnemix.store.SimilarityIndex`
iterates every entry in a namespace (``list_keys`` → ``get`` for each →
cosine). That is an O(N) round-trip scan per query. It's fine up to a few
thousand entries but becomes the dominant latency beyond that.

For production-scale deployments, upgrade to Redis Stack's vector search
(the ``redisearch`` / ``search`` module). The upgrade is backwards-compatible
with this module's storage layout: ``query_embedding`` just needs to be
written as raw FLOAT32 bytes instead of JSON.

**Why VSS is faster:**

* Server-side HNSW or FLAT index over ``query_embedding`` — amortised
  O(log N) KNN instead of O(N).
* Filtering by namespace is a TAG match on the same query.
* One round-trip returns the top-K nearest neighbours directly; no
  per-entry ``HGETALL`` during the scan.

**How to upgrade:**

1. Deploy Redis Stack (stock Redis does not ship the search module).
2. Store embeddings as raw FLOAT32 bytes instead of JSON in the
   ``query_embedding`` field — adjust :func:`_serialize_entry` /
   :func:`_deserialize_entry` accordingly.
3. Create an index once at startup::

       FT.CREATE mnemix-idx ON HASH PREFIX 1 mnemix:entry: SCHEMA
           namespace TAG
           query_embedding VECTOR HNSW 6
               TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE

4. Replace the brute-force scan with a KNN query (pseudo)::

       FT.SEARCH mnemix-idx
           "@namespace:{user_a\\:\\:gpt-4} =>[KNN 1 @query_embedding $vec AS score]"
           PARAMS 2 vec <raw float32 bytes>
           DIALECT 2

**Why VSS isn't the default here:**

* Requires Redis Stack; this module stays portable against stock Redis.
* Index dimension is fixed at ``FT.CREATE`` time and must match the
  embedding model — adds an operational contract between the store and
  the :class:`~mnemix.embedding.EmbeddingEngine`.
* For small caches (≲10K entries), the brute-force path is fine and one
  fewer dependency beats marginal latency gains.

Example:
    Connect and use as any other :class:`CacheStore`::

        >>> import asyncio
        >>> from mnemix.redis_store import RedisStore  # doctest: +SKIP
        >>> async def demo() -> None:  # doctest: +SKIP
        ...     store = RedisStore(url="redis://localhost:6379/0")
        ...     if not await store.ping():
        ...         print("redis down — proxy will run without cache")
        ...         return
        ...     # ... use store as a CacheStore ...
        ...     await store.aclose()
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is a required dep
    RedisError = Exception  # type: ignore[misc,assignment]

from mnemix.store import StoreStats
from mnemix.types import CacheEntry

if TYPE_CHECKING:
    from redis.asyncio import Redis


logger = structlog.get_logger(__name__)


def _serialize_entry(entry: CacheEntry) -> dict[str, str]:
    """Flatten a :class:`CacheEntry` into a Redis hash mapping."""
    return {
        "id": entry.id,
        "namespace": entry.namespace,
        "query_text": entry.query_text,
        "query_embedding": json.dumps(entry.query_embedding),
        "response": json.dumps(entry.response),
        "hit_count": str(entry.hit_count),
        "created_at": entry.created_at.isoformat(),
        "ttl_seconds": "" if entry.ttl_seconds is None else str(entry.ttl_seconds),
        "bypass": "1" if entry.bypass else "0",
    }


def _deserialize_entry(data: dict[str, str]) -> CacheEntry:
    """Rehydrate a :class:`CacheEntry` from a Redis hash mapping.

    Raises:
        KeyError: If required fields are missing.
        ValueError: If fields cannot be decoded (bad JSON, bad datetime).
    """
    ttl_raw = data.get("ttl_seconds", "")
    return CacheEntry(
        id=data["id"],
        namespace=data["namespace"],
        query_text=data["query_text"],
        query_embedding=json.loads(data["query_embedding"]),
        response=json.loads(data["response"]),
        hit_count=int(data.get("hit_count", "0")),
        created_at=datetime.fromisoformat(data["created_at"]),
        ttl_seconds=int(ttl_raw) if ttl_raw else None,
        bypass=data.get("bypass", "0") == "1",
    )


class RedisStore:
    """Redis-backed :class:`~mnemix.store.CacheStore` implementation.

    Conforms to the :class:`~mnemix.store.CacheStore` protocol. All
    methods are safe to call when Redis is unreachable — they log a
    warning and return a cache-miss-shaped default.

    Args:
        client: An already-connected ``redis.asyncio.Redis`` instance (or
            any object with the same method surface, e.g. a test double).
            When provided, the caller retains ownership and
            :meth:`aclose` will not close it.
        url: Connection URL used only when ``client`` is ``None``. The
            default factory passes ``decode_responses=True`` so all
            returned values are ``str``.
        key_prefix: Prefix applied to every Redis key this store owns.
            Use different prefixes to run multiple independent caches on
            one Redis instance.

    Example:
        >>> import asyncio
        >>> from mnemix.redis_store import RedisStore  # doctest: +SKIP
        >>> async def demo() -> None:  # doctest: +SKIP
        ...     store = RedisStore()
        ...     # inject into mnemix.proxy.create_app(store=store)
    """

    def __init__(
        self,
        client: Redis | None = None,
        *,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "mnemix",
    ) -> None:
        """Create a store connected to ``client`` or, failing that, to ``url``."""
        self._owns_client = client is None
        if client is None:
            from redis.asyncio import Redis as _Redis

            client = _Redis.from_url(url, decode_responses=True)
        # Redis' type stubs declare every async method as ``Awaitable[T] | T``,
        # which defeats mypy on ``await`` sites. Typing the client as ``Any``
        # keeps the module's public types strong (CacheStore protocol) without
        # littering the code with casts on every call.
        self._client: Any = client
        self._prefix = key_prefix

    def _entry_key(self, namespace: str, key: str) -> str:
        return f"{self._prefix}:entry:{namespace}:{key}"

    def _ns_set_key(self, namespace: str) -> str:
        return f"{self._prefix}:ns:{namespace}"

    def _all_ns_key(self) -> str:
        return f"{self._prefix}:namespaces"

    @staticmethod
    def _as_str(value: Any) -> str:
        """Accept either ``str`` or ``bytes`` from Redis and return ``str``."""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _decode_hash(raw: dict[Any, Any]) -> dict[str, str]:
        """Coerce an ``HGETALL`` response to ``dict[str, str]``."""
        return {RedisStore._as_str(k): RedisStore._as_str(v) for k, v in raw.items()}

    @staticmethod
    def _decode_set(raw: Any) -> set[str]:
        """Coerce an ``SMEMBERS`` response to ``set[str]``."""
        return {RedisStore._as_str(x) for x in raw}

    async def ping(self) -> bool:
        """Return True when Redis is reachable.

        Never raises — use this to gate a best-effort fallback to another
        store at startup.
        """
        try:
            return bool(await self._client.ping())
        except RedisError as err:
            logger.warning("redis_ping_failed", error=str(err))
            return False

    async def get(self, namespace: str, key: str) -> CacheEntry | None:
        """Return the entry at ``(namespace, key)`` or ``None``.

        Also opportunistically removes the key from the namespace index
        when the underlying hash has expired, keeping the index
        eventually consistent with the keyspace.
        """
        entry_key = self._entry_key(namespace, key)
        try:
            raw = await self._client.hgetall(entry_key)
        except RedisError as err:
            logger.warning("redis_get_failed", error=str(err), key=entry_key)
            return None
        if not raw:
            # TTL'd out or never existed — prune the stale index entry.
            with contextlib.suppress(RedisError):
                await self._client.srem(self._ns_set_key(namespace), key)
            return None
        try:
            return _deserialize_entry(self._decode_hash(raw))
        except (KeyError, ValueError) as err:
            logger.warning("redis_deserialize_failed", error=str(err), key=entry_key)
            return None

    async def set(self, entry: CacheEntry) -> None:
        """Insert or overwrite ``entry``. Applies TTL via native EXPIRE."""
        entry_key = self._entry_key(entry.namespace, entry.id)
        ns_set = self._ns_set_key(entry.namespace)
        all_ns = self._all_ns_key()
        mapping = _serialize_entry(entry)
        try:
            await self._client.hset(entry_key, mapping=mapping)
            if entry.ttl_seconds is not None:
                await self._client.expire(entry_key, entry.ttl_seconds)
            await self._client.sadd(ns_set, entry.id)
            await self._client.sadd(all_ns, entry.namespace)
        except RedisError as err:
            logger.warning("redis_set_failed", error=str(err), key=entry_key)

    async def delete(self, namespace: str, key: str) -> bool:
        """Remove an entry; return True when something was deleted."""
        entry_key = self._entry_key(namespace, key)
        try:
            removed = int(await self._client.delete(entry_key))
            await self._client.srem(self._ns_set_key(namespace), key)
        except RedisError as err:
            logger.warning("redis_delete_failed", error=str(err), key=entry_key)
            return False
        return removed > 0

    async def list_keys(self, namespace: str) -> list[str]:
        """Return entry IDs known in ``namespace``.

        The result comes from the index set and may include IDs whose
        hashes have since expired. :class:`~mnemix.store.SimilarityIndex`
        already skips ``None`` returns from :meth:`get`, so this is safe
        in practice.
        """
        try:
            raw = await self._client.smembers(self._ns_set_key(namespace))
        except RedisError as err:
            logger.warning("redis_list_failed", error=str(err), namespace=namespace)
            return []
        return list(self._decode_set(raw))

    async def clear(self, namespace: str | None = None) -> int:
        """Drop entries. Returns the number of entries removed.

        Args:
            namespace: If provided, clears only that namespace; otherwise
                clears every namespace this store knows about.
        """
        try:
            if namespace is None:
                ns_raw = await self._client.smembers(self._all_ns_key())
                namespaces = self._decode_set(ns_raw)
                count = 0
                for ns in namespaces:
                    count += await self._clear_namespace(ns)
                await self._client.delete(self._all_ns_key())
                return count
            return await self._clear_namespace(namespace)
        except RedisError as err:
            logger.warning("redis_clear_failed", error=str(err))
            return 0

    async def _clear_namespace(self, namespace: str) -> int:
        ids_raw = await self._client.smembers(self._ns_set_key(namespace))
        ids = self._decode_set(ids_raw)
        count = 0
        for entry_id in ids:
            removed = int(
                await self._client.delete(self._entry_key(namespace, entry_id)),
            )
            count += removed
        await self._client.delete(self._ns_set_key(namespace))
        await self._client.srem(self._all_ns_key(), namespace)
        return count

    async def stats(self) -> StoreStats:
        """Return per-namespace and total entry counts.

        Counts come from the index sets (SCARD). Entries that have TTL'd
        out but haven't been ``get``-pruned yet may be overcounted; this
        is documented as an eventually-consistent trade-off.
        """
        per_ns: dict[str, int] = {}
        try:
            ns_raw = await self._client.smembers(self._all_ns_key())
            namespaces = self._decode_set(ns_raw)
            total = 0
            for ns in namespaces:
                count = int(await self._client.scard(self._ns_set_key(ns)))
                if count > 0:
                    per_ns[ns] = count
                    total += count
            return StoreStats(total_entries=total, entries_per_namespace=per_ns)
        except RedisError as err:
            logger.warning("redis_stats_failed", error=str(err))
            return StoreStats(total_entries=0, entries_per_namespace={})

    async def aclose(self) -> None:
        """Close the underlying connection pool if this store owns it."""
        if self._owns_client:
            try:
                await self._client.aclose()
            except RedisError as err:
                logger.warning("redis_close_failed", error=str(err))


__all__ = ["RedisStore"]
