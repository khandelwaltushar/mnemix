"""HTTP proxy for Mnemix.

Exposes an OpenAI- and Anthropic-compatible FastAPI app that intercepts
chat completion requests, serves semantically similar cached responses when
possible, and forwards the rest upstream.

A client using the OpenAI SDK points at this proxy by changing only
``base_url``; no other code changes are needed.

Example:
    Run a standalone proxy::

        >>> from mnemix.proxy import create_app
        >>> app = create_app()  # doctest: +SKIP
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)  # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mnemix.bypass import BypassClassifier, RuleBasedBypass
from mnemix.embedding import EmbeddingEngine, get_sentence_transformer_engine
from mnemix.store import CacheStore, InMemoryStore, SimilarityIndex
from mnemix.types import CacheConfig, CacheEntry, MetricsSnapshot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = structlog.get_logger(__name__)


_PRICING_PER_1K_TOKENS: dict[str, float] = {
    "gpt-4o-mini": 0.0003,
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.01,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.0005,
    "claude-3-5-sonnet": 0.003,
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "claude-sonnet-4": 0.003,
    "claude-opus-4": 0.015,
    "claude-haiku-4": 0.00025,
}
_DEFAULT_PRICE_PER_1K_TOKENS = 0.005


def estimate_cost_usd(model: str, tokens: int) -> float:
    """Estimate the upstream cost of a response, in USD.

    Looks up ``model`` against an approximate per-1K-token pricing table
    (longest prefix wins). Unknown models fall back to a flat default.

    Args:
        model: Provider model identifier, e.g. ``"gpt-4o"`` or
            ``"claude-3-5-sonnet-20241022"``.
        tokens: Total tokens in the cached response (prompt + completion).

    Returns:
        Estimated USD cost, non-negative.

    Example:
        >>> estimate_cost_usd("gpt-4o", 1000)
        0.005
    """
    if tokens <= 0:
        return 0.0
    price = _DEFAULT_PRICE_PER_1K_TOKENS
    best_match_len = -1
    for prefix, p in _PRICING_PER_1K_TOKENS.items():
        if model.startswith(prefix) and len(prefix) > best_match_len:
            price = p
            best_match_len = len(prefix)
    return (tokens / 1000.0) * price


class MetricsTracker:
    """Running counters for cache performance.

    Tracks total requests, hits/misses, cumulative similarity on hits, and
    estimated token + dollar savings. Produces an immutable
    :class:`MetricsSnapshot` for the ``/metrics`` endpoint.

    Example:
        >>> t = MetricsTracker()
        >>> t.record_miss()
        >>> t.record_hit(similarity=0.98, tokens_saved=150, cost_saved=0.0008)
        >>> t.snapshot().total_requests
        2
    """

    def __init__(self) -> None:
        """Initialize all counters at zero."""
        self._total_requests = 0
        self._hits = 0
        self._misses = 0
        self._similarity_sum = 0.0
        self._tokens_saved = 0
        self._cost_saved = 0.0

    def record_hit(self, similarity: float, tokens_saved: int, cost_saved: float) -> None:
        """Record a cache hit."""
        self._total_requests += 1
        self._hits += 1
        self._similarity_sum += similarity
        self._tokens_saved += max(0, tokens_saved)
        self._cost_saved += max(0.0, cost_saved)

    def record_miss(self) -> None:
        """Record a cache miss (or a bypass)."""
        self._total_requests += 1
        self._misses += 1

    def snapshot(self) -> MetricsSnapshot:
        """Produce an immutable view of the current counters."""
        hit_rate = self._hits / self._total_requests if self._total_requests else 0.0
        avg_sim = (self._similarity_sum / self._hits) if self._hits else 0.0
        # Clamp against float drift; the snapshot's validators reject >1.0.
        avg_sim = max(0.0, min(1.0, avg_sim))
        hit_rate = max(0.0, min(1.0, hit_rate))
        return MetricsSnapshot(
            total_requests=self._total_requests,
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
            avg_similarity_on_hit=avg_sim,
            estimated_tokens_saved=self._tokens_saved,
            estimated_cost_saved_usd=round(self._cost_saved, 6),
        )


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI/Anthropic ``content`` field into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def serialize_openai_query(body: dict[str, Any]) -> str:
    """Canonicalize an OpenAI chat-completions body into a cacheable string."""
    parts: list[str] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            parts.append(f"{role}: {_content_to_text(m.get('content'))}")
    return "\n".join(parts)


def serialize_anthropic_query(body: dict[str, Any]) -> str:
    """Canonicalize an Anthropic messages body into a cacheable string."""
    parts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        parts.append(f"system: {system}")
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(f"system: {text}")
    messages = body.get("messages")
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            parts.append(f"{role}: {_content_to_text(m.get('content'))}")
    return "\n".join(parts)


def _total_tokens(response: dict[str, Any]) -> int:
    """Extract total tokens from an OpenAI- or Anthropic-shaped response."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens + output_tokens
    return 0


def _entry_id(namespace: str, model: str, query_text: str) -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(query_text.encode("utf-8"))
    return h.hexdigest()[:32]


def _forwarded_headers(request: Request) -> dict[str, str]:
    """Return request headers suitable for forwarding upstream."""
    skip = {"host", "content-length", "connection", "accept-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _response_from_upstream(upstream: httpx.Response) -> Response:
    """Wrap an upstream httpx response as a Starlette Response."""
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def create_app(
    config: CacheConfig | None = None,
    *,
    store: CacheStore | None = None,
    embedding_engine: EmbeddingEngine | None = None,
    bypass_classifier: BypassClassifier | None = None,
    http_client: httpx.AsyncClient | None = None,
    metrics: MetricsTracker | None = None,
) -> FastAPI:
    """Build a configured FastAPI proxy app.

    All collaborators have sensible defaults so ``create_app()`` with no
    arguments produces a working proxy. Every collaborator is also
    injectable for tests — pass a stub embedding engine or an
    :class:`httpx.AsyncClient` wired to a :class:`httpx.MockTransport` to
    avoid real network calls.

    Args:
        config: Cache configuration. Defaults to :class:`CacheConfig()`.
        store: Cache backend. Defaults to a fresh :class:`InMemoryStore`
            sized to ``config.max_cache_size``.
        embedding_engine: Embedding provider. Defaults to the local
            SentenceTransformer engine with ``config.embedding_model``.
        bypass_classifier: Query-level bypass. Defaults to
            :class:`RuleBasedBypass` with the built-in patterns.
        http_client: Upstream HTTP client. Defaults to a fresh
            :class:`httpx.AsyncClient` with the configured timeout; the
            app then owns and closes it on shutdown. When injected, the
            caller retains ownership.
        metrics: Metrics tracker. Defaults to a fresh one.

    Returns:
        A FastAPI app exposing ``/v1/chat/completions``, ``/v1/messages``,
        ``/metrics``, and ``/health``.

    Example:
        >>> from mnemix.proxy import create_app
        >>> app = create_app()  # doctest: +SKIP
    """
    cfg = config or CacheConfig()
    cache_store: CacheStore = store or InMemoryStore(max_size=cfg.max_cache_size)
    engine: EmbeddingEngine = embedding_engine or get_sentence_transformer_engine(
        cfg.embedding_model,
    )
    classifier: BypassClassifier = bypass_classifier or RuleBasedBypass()
    index = SimilarityIndex(cache_store, threshold=cfg.similarity_threshold)
    tracker = metrics or MetricsTracker()
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.upstream_timeout_seconds),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owned_client:
                await client.aclose()

    app = FastAPI(
        title="Mnemix",
        version="0.1.0",
        description="Semantic caching proxy for LLM APIs.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _cache_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        info = getattr(request.state, "mnemix_cache_info", None)
        if isinstance(info, dict):
            hit = bool(info.get("hit"))
            response.headers["X-Mnemix-Cache"] = "HIT" if hit else "MISS"
            if hit:
                sim = info.get("similarity")
                if isinstance(sim, int | float):
                    response.headers["X-Mnemix-Similarity"] = f"{float(sim):.4f}"
        return response

    async def _handle(
        request: Request,
        *,
        upstream_url: str,
        extract_query: Callable[[dict[str, Any]], str],
    ) -> Response:
        body_bytes = await request.body()
        try:
            parsed = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(parsed, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        user_ns = request.headers.get(cfg.namespace_header, cfg.default_namespace)
        model = str(parsed.get("model", "unknown"))
        namespace = f"{user_ns}::{model}"
        stream = bool(parsed.get("stream", False))
        query_text = extract_query(parsed)

        if stream:
            # Streaming passthrough — never cached.
            tracker.record_miss()
            request.state.mnemix_cache_info = {"hit": False}
            return await _stream_passthrough(
                request=request,
                upstream_url=upstream_url,
                body_bytes=body_bytes,
                client=client,
            )

        if not query_text.strip() or classifier.should_bypass(query_text):
            tracker.record_miss()
            request.state.mnemix_cache_info = {"hit": False}
            upstream = await client.post(
                upstream_url,
                content=body_bytes,
                headers=_forwarded_headers(request),
            )
            return _response_from_upstream(upstream)

        embedding = await engine.embed(query_text)
        result = await index.search(embedding, namespace=namespace)

        if result.hit and result.entry is not None and result.similarity_score is not None:
            bumped = result.entry.model_copy(
                update={"hit_count": result.entry.hit_count + 1},
            )
            await cache_store.set(bumped)
            tokens = _total_tokens(result.entry.response)
            cost = estimate_cost_usd(model, tokens)
            tracker.record_hit(
                similarity=result.similarity_score,
                tokens_saved=tokens,
                cost_saved=cost,
            )
            request.state.mnemix_cache_info = {
                "hit": True,
                "similarity": result.similarity_score,
            }
            logger.info(
                "cache_hit",
                namespace=namespace,
                similarity=result.similarity_score,
                tokens_saved=tokens,
            )
            return JSONResponse(content=result.entry.response, status_code=200)

        tracker.record_miss()
        request.state.mnemix_cache_info = {"hit": False}
        upstream = await client.post(
            upstream_url,
            content=body_bytes,
            headers=_forwarded_headers(request),
        )

        if upstream.status_code == 200:
            try:
                resp_json = upstream.json()
            except (ValueError, json.JSONDecodeError):
                resp_json = None
            if isinstance(resp_json, dict):
                entry = CacheEntry(
                    id=_entry_id(namespace, model, query_text),
                    namespace=namespace,
                    query_text=query_text,
                    query_embedding=embedding,
                    response=resp_json,
                    ttl_seconds=cfg.default_ttl,
                )
                await cache_store.set(entry)
                logger.info(
                    "cache_store",
                    namespace=namespace,
                    query_len=len(query_text),
                )

        return _response_from_upstream(upstream)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/metrics", response_model=MetricsSnapshot)
    async def metrics_endpoint() -> MetricsSnapshot:
        """Return a point-in-time :class:`MetricsSnapshot`."""
        return tracker.snapshot()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """OpenAI-compatible chat completions endpoint."""
        return await _handle(
            request,
            upstream_url=f"{cfg.openai_base_url.rstrip('/')}/v1/chat/completions",
            extract_query=serialize_openai_query,
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        """Anthropic-compatible messages endpoint."""
        return await _handle(
            request,
            upstream_url=f"{cfg.anthropic_base_url.rstrip('/')}/v1/messages",
            extract_query=serialize_anthropic_query,
        )

    return app


async def _stream_passthrough(
    *,
    request: Request,
    upstream_url: str,
    body_bytes: bytes,
    client: httpx.AsyncClient,
) -> StreamingResponse:
    """Pipe an upstream streamed response back to the client unchanged."""
    upstream_req = client.build_request(
        "POST",
        upstream_url,
        content=body_bytes,
        headers=_forwarded_headers(request),
    )
    upstream = await client.send(upstream_req, stream=True)

    async def iter_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    media_type = upstream.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        iter_body(),
        status_code=upstream.status_code,
        media_type=media_type,
    )


__all__ = [
    "MetricsTracker",
    "create_app",
    "estimate_cost_usd",
    "serialize_anthropic_query",
    "serialize_openai_query",
]
