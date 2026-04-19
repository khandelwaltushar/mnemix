"""Tests for mnemix.proxy: end-to-end with a mocked httpx upstream."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import httpx
import pytest

from mnemix.bypass import AlwaysBypass, NeverBypass
from mnemix.proxy import (
    MetricsTracker,
    create_app,
    estimate_cost_usd,
    serialize_anthropic_query,
    serialize_openai_query,
)
from mnemix.store import InMemoryStore
from mnemix.types import CacheConfig


class StubEmbeddingEngine:
    """Deterministic engine: same text → same vector; different texts → different."""

    dimension = 16

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.dimension]
        vec = [(b / 255.0) - 0.5 for b in digest]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return [0.0] * self.dimension
        return [x / norm for x in vec]


def _openai_response(content: str = "hi there", model: str = "gpt-4o") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response(content: str = "hi there") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }


class UpstreamRecorder:
    """httpx.MockTransport handler that records calls and returns canned JSON."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self._responses = responses or {
            "/v1/chat/completions": _openai_response(),
            "/v1/messages": _anthropic_response(),
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path in self._responses:
            return httpx.Response(200, json=self._responses[path])
        return httpx.Response(404, json={"error": f"no mock for {path}"})


def _build_app(
    *,
    upstream: UpstreamRecorder | None = None,
    bypass_classifier: Any = None,
    store: InMemoryStore | None = None,
    threshold: float = 0.92,
) -> tuple[Any, UpstreamRecorder]:
    upstream = upstream or UpstreamRecorder()
    transport = httpx.MockTransport(upstream)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    config = CacheConfig(
        similarity_threshold=threshold,
        openai_base_url="https://fake-openai.test",
        anthropic_base_url="https://fake-anthropic.test",
    )
    app = create_app(
        config,
        store=store or InMemoryStore(max_size=100),
        embedding_engine=StubEmbeddingEngine(),
        bypass_classifier=bypass_classifier or NeverBypass(),
        http_client=client,
    )
    return app, upstream


async def _async_client(app: Any) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class TestSerializers:
    def test_openai_basic(self) -> None:
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert "user: hello" in serialize_openai_query(body)

    def test_openai_multiturn(self) -> None:
        body = {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ],
        }
        out = serialize_openai_query(body)
        assert "system: be helpful" in out
        assert "user: hi" in out

    def test_openai_content_blocks(self) -> None:
        content = [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]
        body = {"messages": [{"role": "user", "content": content}]}
        assert "part1" in serialize_openai_query(body)
        assert "part2" in serialize_openai_query(body)

    def test_anthropic_with_system_string(self) -> None:
        body = {"system": "s", "messages": [{"role": "user", "content": "q"}]}
        out = serialize_anthropic_query(body)
        assert "system: s" in out
        assert "user: q" in out

    def test_anthropic_with_system_blocks(self) -> None:
        body = {
            "system": [{"type": "text", "text": "sys-a"}, {"type": "text", "text": "sys-b"}],
            "messages": [{"role": "user", "content": "q"}],
        }
        out = serialize_anthropic_query(body)
        assert "system: sys-a" in out
        assert "system: sys-b" in out

    def test_handles_missing_fields(self) -> None:
        assert serialize_openai_query({}) == ""
        assert serialize_anthropic_query({}) == ""


class TestEstimateCost:
    def test_known_model(self) -> None:
        assert estimate_cost_usd("gpt-4o", 1000) == pytest.approx(0.005)

    def test_claude_prefix_match(self) -> None:
        assert estimate_cost_usd("claude-3-5-sonnet-20241022", 1000) == pytest.approx(0.003)

    def test_unknown_model_uses_default(self) -> None:
        assert estimate_cost_usd("mystery-model", 1000) == pytest.approx(0.005)

    def test_zero_tokens(self) -> None:
        assert estimate_cost_usd("gpt-4o", 0) == 0.0

    def test_negative_tokens_clamped(self) -> None:
        assert estimate_cost_usd("gpt-4o", -100) == 0.0

    def test_longest_prefix_wins(self) -> None:
        # "gpt-4o-mini" must beat "gpt-4o" when the model is "gpt-4o-mini-2024"
        assert estimate_cost_usd("gpt-4o-mini-2024-07-18", 1000) == pytest.approx(0.0003)


class TestMetricsTracker:
    def test_empty(self) -> None:
        snap = MetricsTracker().snapshot()
        assert snap.total_requests == 0
        assert snap.hits == 0
        assert snap.hit_rate == 0.0
        assert snap.avg_similarity_on_hit == 0.0

    def test_after_hits_and_misses(self) -> None:
        t = MetricsTracker()
        t.record_miss()
        t.record_hit(similarity=0.95, tokens_saved=100, cost_saved=0.0005)
        t.record_hit(similarity=0.99, tokens_saved=50, cost_saved=0.00025)
        snap = t.snapshot()
        assert snap.total_requests == 3
        assert snap.hits == 2
        assert snap.misses == 1
        assert snap.hit_rate == pytest.approx(2 / 3)
        assert snap.avg_similarity_on_hit == pytest.approx(0.97)
        assert snap.estimated_tokens_saved == 150
        assert snap.estimated_cost_saved_usd == pytest.approx(0.00075)


class TestHealth:
    async def test_ok(self) -> None:
        app, _ = _build_app()
        async with await _async_client(app) as ac:
            r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestMetricsEndpoint:
    async def test_empty_snapshot(self) -> None:
        app, _ = _build_app()
        async with await _async_client(app) as ac:
            r = await ac.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["total_requests"] == 0
        assert body["hits"] == 0
        assert body["misses"] == 0


class TestChatCompletions:
    async def test_miss_calls_upstream_and_returns_response(self) -> None:
        app, upstream = _build_app()
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            r = await ac.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert len(upstream.calls) == 1
        assert r.headers["x-mnemix-cache"] == "MISS"
        assert "x-mnemix-similarity" not in r.headers
        # Response shape preserved
        assert r.json()["id"] == "chatcmpl-test"

    async def test_identical_second_call_is_cache_hit(self) -> None:
        app, upstream = _build_app()
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/chat/completions", json=payload)
            r2 = await ac.post("/v1/chat/completions", json=payload)
        assert r1.headers["x-mnemix-cache"] == "MISS"
        assert r2.headers["x-mnemix-cache"] == "HIT"
        assert "x-mnemix-similarity" in r2.headers
        # Upstream was called exactly once (for the miss)
        assert len(upstream.calls) == 1
        # Cached body returned
        assert r2.json()["id"] == r1.json()["id"]

    async def test_different_query_is_cache_miss(self) -> None:
        app, upstream = _build_app()
        async with await _async_client(app) as ac:
            await ac.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            )
            await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "tell me about sql query optimization"},
                    ],
                },
            )
        assert len(upstream.calls) == 2

    async def test_metrics_update_after_hit(self) -> None:
        app, _ = _build_app()
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            await ac.post("/v1/chat/completions", json=payload)
            await ac.post("/v1/chat/completions", json=payload)
            m = await ac.get("/metrics")
        body = m.json()
        assert body["total_requests"] == 2
        assert body["hits"] == 1
        assert body["misses"] == 1
        assert body["hit_rate"] == pytest.approx(0.5)
        assert body["estimated_tokens_saved"] == 15  # from the mocked usage
        assert body["estimated_cost_saved_usd"] > 0.0

    async def test_authorization_header_forwarded(self) -> None:
        app, upstream = _build_app()
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            await ac.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer sk-test"},
            )
        assert upstream.calls[0].headers.get("authorization") == "Bearer sk-test"

    async def test_namespace_isolation_between_tenants(self) -> None:
        app, upstream = _build_app()
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            await ac.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Mnemix-Namespace": "tenant-a"},
            )
            r_b = await ac.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Mnemix-Namespace": "tenant-b"},
            )
        # tenant-b should NOT hit tenant-a's cache
        assert r_b.headers["x-mnemix-cache"] == "MISS"
        assert len(upstream.calls) == 2

    async def test_bypass_query_forwards_and_does_not_cache(self) -> None:
        app, upstream = _build_app(bypass_classifier=AlwaysBypass())
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "anything"}]}
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/chat/completions", json=payload)
            r2 = await ac.post("/v1/chat/completions", json=payload)
        assert r1.headers["x-mnemix-cache"] == "MISS"
        assert r2.headers["x-mnemix-cache"] == "MISS"
        assert len(upstream.calls) == 2

    async def test_different_models_do_not_share_cache(self) -> None:
        app, upstream = _build_app()
        msgs = [{"role": "user", "content": "hello"}]
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": msgs})
            r2 = await ac.post(
                "/v1/chat/completions",
                json={"model": "gpt-4-turbo", "messages": msgs},
            )
        assert r1.headers["x-mnemix-cache"] == "MISS"
        assert r2.headers["x-mnemix-cache"] == "MISS"
        assert len(upstream.calls) == 2

    async def test_invalid_json_returns_400(self) -> None:
        app, upstream = _build_app()
        async with await _async_client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
        assert r.status_code == 400
        assert len(upstream.calls) == 0

    async def test_non_object_body_returns_400(self) -> None:
        app, _ = _build_app()
        async with await _async_client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                content=json.dumps([1, 2, 3]).encode(),
                headers={"Content-Type": "application/json"},
            )
        assert r.status_code == 400

    async def test_upstream_error_is_not_cached(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "upstream down"})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport, timeout=5.0)
        app = create_app(
            CacheConfig(openai_base_url="https://fake.test"),
            store=InMemoryStore(max_size=10),
            embedding_engine=StubEmbeddingEngine(),
            bypass_classifier=NeverBypass(),
            http_client=client,
        )
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/chat/completions", json=payload)
            r2 = await ac.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 500
        # Second call should still MISS (nothing was cached)
        assert r2.headers["x-mnemix-cache"] == "MISS"


class TestAnthropicMessages:
    async def test_miss_then_hit(self) -> None:
        app, upstream = _build_app()
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
        }
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/messages", json=payload)
            r2 = await ac.post("/v1/messages", json=payload)
        assert r1.headers["x-mnemix-cache"] == "MISS"
        assert r2.headers["x-mnemix-cache"] == "HIT"
        assert len(upstream.calls) == 1
        assert r1.json()["id"] == "msg_test"

    async def test_hit_saves_anthropic_tokens(self) -> None:
        app, _ = _build_app()
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
        }
        async with await _async_client(app) as ac:
            await ac.post("/v1/messages", json=payload)
            await ac.post("/v1/messages", json=payload)
            m = await ac.get("/metrics")
        body = m.json()
        # 8 input + 4 output = 12 tokens saved on the hit
        assert body["estimated_tokens_saved"] == 12


class TestSimilarityThreshold:
    # Note: the stub engine is hash-based so "similar" texts don't produce
    # similar vectors. These tests exercise the threshold wiring with
    # identical queries (always 1.0 similarity) and unrelated queries
    # (near-zero similarity under the stub).
    async def test_identical_query_hits_even_at_high_threshold(self) -> None:
        app, upstream = _build_app(threshold=0.999999)
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        async with await _async_client(app) as ac:
            r1 = await ac.post("/v1/chat/completions", json=payload)
            r2 = await ac.post("/v1/chat/completions", json=payload)
        assert r1.headers["x-mnemix-cache"] == "MISS"
        assert r2.headers["x-mnemix-cache"] == "HIT"
        # Similarity header rounds to 1.0000
        assert r2.headers["x-mnemix-similarity"].startswith("1.0000")
        assert len(upstream.calls) == 1

    async def test_unrelated_queries_miss_under_strict_threshold(self) -> None:
        app, upstream = _build_app(threshold=0.999999)
        async with await _async_client(app) as ac:
            await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hello world"}],
                },
            )
            r2 = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "tell me about quantum entanglement"},
                    ],
                },
            )
        assert r2.headers["x-mnemix-cache"] == "MISS"
        assert len(upstream.calls) == 2
