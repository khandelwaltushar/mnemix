"""Tests for mnemix.redis_store: RedisStore against a fake async Redis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from redis.exceptions import ConnectionError as RedisConnectionError

if TYPE_CHECKING:
    import pytest

from mnemix.redis_store import (
    RedisStore,
    _deserialize_entry,
    _serialize_entry,
)
from mnemix.store import CacheStore, SimilarityIndex, StoreStats
from mnemix.types import CacheEntry


class FakeAsyncRedis:
    """In-memory stand-in for ``redis.asyncio.Redis`` with just the surface we use."""

    def __init__(self, *, broken: bool = False) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.broken = broken
        self.aclose_calls = 0

    def _check(self) -> None:
        if self.broken:
            raise RedisConnectionError("fake redis is broken")

    async def ping(self) -> bool:
        self._check()
        return True

    async def hset(
        self,
        name: str,
        *,
        mapping: dict[str, Any] | None = None,
    ) -> int:
        self._check()
        stored = {str(k): str(v) for k, v in (mapping or {}).items()}
        existing = self.hashes.setdefault(name, {})
        added = sum(1 for k in stored if k not in existing)
        existing.update(stored)
        return added

    async def hgetall(self, name: str) -> dict[str, str]:
        self._check()
        return dict(self.hashes.get(name, {}))

    async def delete(self, *names: str) -> int:
        self._check()
        count = 0
        for n in names:
            if n in self.hashes:
                del self.hashes[n]
                count += 1
            if n in self.sets:
                del self.sets[n]
        return count

    async def sadd(self, name: str, *values: str) -> int:
        self._check()
        s = self.sets.setdefault(name, set())
        added = 0
        for v in values:
            vs = str(v)
            if vs not in s:
                s.add(vs)
                added += 1
        return added

    async def srem(self, name: str, *values: str) -> int:
        self._check()
        s = self.sets.get(name)
        if not s:
            return 0
        removed = 0
        for v in values:
            vs = str(v)
            if vs in s:
                s.remove(vs)
                removed += 1
        if not s:
            del self.sets[name]
        return removed

    async def smembers(self, name: str) -> set[str]:
        self._check()
        return set(self.sets.get(name, set()))

    async def scard(self, name: str) -> int:
        self._check()
        return len(self.sets.get(name, set()))

    async def expire(self, name: str, seconds: int) -> bool:
        self._check()
        _ = seconds
        return name in self.hashes

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _entry(
    *,
    id: str = "q1",
    namespace: str = "ns",
    embedding: list[float] | None = None,
    response: dict[str, Any] | None = None,
    bypass: bool = False,
    hit_count: int = 0,
    ttl_seconds: int | None = None,
    created_at: datetime | None = None,
) -> CacheEntry:
    return CacheEntry(
        id=id,
        namespace=namespace,
        query_text=f"text for {id}",
        query_embedding=embedding if embedding is not None else [1.0, 0.0, 0.0],
        response=response if response is not None else {"text": f"resp {id}"},
        hit_count=hit_count,
        bypass=bypass,
        ttl_seconds=ttl_seconds,
        created_at=created_at or datetime.now(tz=UTC),
    )


class TestSerialization:
    def test_roundtrip_basic(self) -> None:
        e = _entry(id="abc", embedding=[0.1, 0.2, 0.3], response={"n": 1, "s": "hi"})
        blob = _serialize_entry(e)
        assert all(isinstance(v, str) for v in blob.values())
        back = _deserialize_entry(blob)
        assert back == e

    def test_roundtrip_preserves_ttl_and_bypass(self) -> None:
        e = _entry(id="abc", ttl_seconds=60, bypass=True, hit_count=7)
        back = _deserialize_entry(_serialize_entry(e))
        assert back.ttl_seconds == 60
        assert back.bypass is True
        assert back.hit_count == 7

    def test_none_ttl_roundtrips(self) -> None:
        e = _entry(id="abc", ttl_seconds=None)
        blob = _serialize_entry(e)
        assert blob["ttl_seconds"] == ""
        back = _deserialize_entry(blob)
        assert back.ttl_seconds is None

    def test_decodes_bytes_keys_and_values(self) -> None:
        # Simulate decode_responses=False — values arrive as bytes.
        e = _entry(id="abc")
        blob = _serialize_entry(e)
        bytes_blob: dict[Any, Any] = {k.encode(): v.encode() for k, v in blob.items()}
        decoded = RedisStore._decode_hash(bytes_blob)
        assert _deserialize_entry(decoded) == e


class TestRedisStoreBasics:
    async def test_conforms_to_protocol(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        assert isinstance(store, CacheStore)

    async def test_set_then_get(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)
        entry = _entry(id="q1", namespace="ns")
        await store.set(entry)
        got = await store.get("ns", "q1")
        assert got == entry

    async def test_get_missing_returns_none(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        assert await store.get("ns", "nope") is None

    async def test_get_prunes_index_when_hash_missing(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)
        # Simulate TTL expiry: id still in the namespace set, hash is gone.
        await fake.sadd("mnemix:ns:ns", "ghost")
        assert "ghost" in await fake.smembers("mnemix:ns:ns")
        assert await store.get("ns", "ghost") is None
        assert "ghost" not in await fake.smembers("mnemix:ns:ns")

    async def test_delete(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)
        await store.set(_entry(id="q1", namespace="ns"))
        assert await store.delete("ns", "q1") is True
        assert await store.delete("ns", "q1") is False
        assert await store.get("ns", "q1") is None

    async def test_list_keys_namespace_filtered(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="a"))
        await store.set(_entry(id="q3", namespace="b"))
        assert sorted(await store.list_keys("a")) == ["q1", "q2"]
        assert await store.list_keys("b") == ["q3"]

    async def test_namespace_isolation_on_get(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        await store.set(_entry(id="shared", namespace="user_a"))
        assert await store.get("user_b", "shared") is None

    async def test_clear_one_namespace(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="a"))
        await store.set(_entry(id="q3", namespace="b"))
        removed = await store.clear("a")
        assert removed == 2
        assert await store.list_keys("a") == []
        assert sorted(await store.list_keys("b")) == ["q3"]

    async def test_clear_all(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        for i in range(3):
            await store.set(_entry(id=f"q{i}", namespace="ns"))
        removed = await store.clear()
        assert removed == 3
        assert await store.list_keys("ns") == []

    async def test_stats(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        await store.set(_entry(id="q1", namespace="a"))
        await store.set(_entry(id="q2", namespace="a"))
        await store.set(_entry(id="q3", namespace="b"))
        stats = await store.stats()
        assert isinstance(stats, StoreStats)
        assert stats.total_entries == 3
        assert stats.entries_per_namespace == {"a": 2, "b": 1}

    async def test_set_applies_ttl_via_expire(self) -> None:
        fake = FakeAsyncRedis()
        calls: list[tuple[str, int]] = []
        original = fake.expire

        async def spy(name: str, seconds: int) -> bool:
            calls.append((name, seconds))
            return await original(name, seconds)

        fake.expire = spy  # type: ignore[method-assign]
        store = RedisStore(client=fake)
        await store.set(_entry(id="q1", namespace="ns", ttl_seconds=60))
        assert calls == [("mnemix:entry:ns:q1", 60)]

    async def test_set_without_ttl_does_not_expire(self) -> None:
        fake = FakeAsyncRedis()
        calls: list[tuple[str, int]] = []
        original = fake.expire

        async def spy(name: str, seconds: int) -> bool:
            calls.append((name, seconds))
            return await original(name, seconds)

        fake.expire = spy  # type: ignore[method-assign]
        store = RedisStore(client=fake)
        await store.set(_entry(id="q1", namespace="ns", ttl_seconds=None))
        assert calls == []


class TestRedisStoreKeyPrefix:
    async def test_custom_prefix_namespacing(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake, key_prefix="app1")
        await store.set(_entry(id="q1", namespace="ns"))
        assert "app1:entry:ns:q1" in fake.hashes
        assert "mnemix:entry:ns:q1" not in fake.hashes

    async def test_two_prefixes_isolated(self) -> None:
        fake = FakeAsyncRedis()
        s1 = RedisStore(client=fake, key_prefix="app1")
        s2 = RedisStore(client=fake, key_prefix="app2")
        await s1.set(_entry(id="shared", namespace="ns"))
        assert await s1.get("ns", "shared") is not None
        assert await s2.get("ns", "shared") is None


class TestRedisStoreWithSimilarityIndex:
    async def test_similarity_search_over_redis(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        await store.set(_entry(id="q1", namespace="ns", embedding=[1.0, 0.0, 0.0]))
        await store.set(_entry(id="q2", namespace="ns", embedding=[0.0, 1.0, 0.0]))
        index = SimilarityIndex(store, threshold=0.9)
        result = await index.search([1.0, 0.01, 0.0], namespace="ns")
        assert result.hit is True
        assert result.entry is not None
        assert result.entry.id == "q1"

    async def test_similarity_skips_expired_via_get(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
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
        # SimilarityIndex filters expired entries in-process.
        assert result.hit is False


class TestRedisStoreGracefulFallback:
    async def test_ping_returns_false_when_redis_down(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        assert await store.ping() is False

    async def test_ping_returns_true_when_up(self) -> None:
        store = RedisStore(client=FakeAsyncRedis())
        assert await store.ping() is True

    async def test_get_returns_none_on_connection_error(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        assert await store.get("ns", "q1") is None

    async def test_set_is_noop_on_connection_error(self) -> None:
        fake = FakeAsyncRedis(broken=True)
        store = RedisStore(client=fake)
        # Must not raise.
        await store.set(_entry(id="q1", namespace="ns"))
        # Fake's state should be unchanged since every op errored.
        assert fake.hashes == {}

    async def test_delete_returns_false_on_connection_error(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        assert await store.delete("ns", "q1") is False

    async def test_list_keys_returns_empty_on_connection_error(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        assert await store.list_keys("ns") == []

    async def test_clear_returns_zero_on_connection_error(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        assert await store.clear("ns") == 0
        assert await store.clear() == 0

    async def test_stats_returns_empty_on_connection_error(self) -> None:
        store = RedisStore(client=FakeAsyncRedis(broken=True))
        stats = await store.stats()
        assert stats.total_entries == 0
        assert stats.entries_per_namespace == {}

    async def test_corrupted_hash_returns_none(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)
        # Write garbage directly under an entry key.
        fake.hashes["mnemix:entry:ns:bad"] = {"id": "bad", "query_embedding": "not-json"}
        await fake.sadd("mnemix:ns:ns", "bad")
        assert await store.get("ns", "bad") is None

    async def test_get_still_works_when_index_srem_fails(self) -> None:
        # Break srem but leave everything else working by flipping the flag
        # only around the specific call. Easiest: wrap srem on the fake.
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)

        async def broken_srem(name: str, *values: str) -> int:
            _ = name, values
            raise RedisConnectionError("boom")

        fake.srem = broken_srem  # type: ignore[method-assign]
        # Missing hash — get will try to prune index and that call raises,
        # but the overall get must still return None (not propagate).
        assert await store.get("ns", "nope") is None


class TestRedisStoreLifecycle:
    async def test_aclose_closes_owned_client(self) -> None:
        # Monkeypatch the factory so we don't need a live Redis.
        fake = FakeAsyncRedis()
        store = RedisStore(client=None, url="redis://invalid/0")
        # Replace the client with our fake but keep owns_client=True.
        store._client = fake
        await store.aclose()
        assert fake.aclose_calls == 1

    async def test_aclose_leaves_injected_client_alone(self) -> None:
        fake = FakeAsyncRedis()
        store = RedisStore(client=fake)
        await store.aclose()
        assert fake.aclose_calls == 0

    async def test_aclose_swallows_redis_errors(self) -> None:
        fake = FakeAsyncRedis()

        async def bad_close() -> None:
            raise RedisConnectionError("boom")

        fake.aclose = bad_close  # type: ignore[method-assign]
        store = RedisStore(client=None, url="redis://invalid/0")
        store._client = fake
        # Should not raise.
        await store.aclose()


class TestRedisStoreDefaultFactory:
    def test_from_url_constructs_real_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Verify the default path uses redis.asyncio.Redis.from_url with
        # decode_responses=True — without requiring a live server.
        captured: dict[str, Any] = {}

        import redis.asyncio as redis_asyncio

        real_from_url = redis_asyncio.Redis.from_url

        def fake_from_url(url: str, **kwargs: Any) -> Any:
            captured["url"] = url
            captured["kwargs"] = kwargs
            # Return a real client object — we never talk to it.
            return real_from_url(url, **kwargs)

        monkeypatch.setattr(redis_asyncio.Redis, "from_url", fake_from_url)
        store = RedisStore(url="redis://example:6379/2")
        assert captured["url"] == "redis://example:6379/2"
        assert captured["kwargs"].get("decode_responses") is True
        assert store._owns_client is True
