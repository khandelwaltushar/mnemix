"""Demonstrate per-tenant cache isolation via the namespace header.

Two users (``alice`` and ``bob``) ask the *same question* through the
same Mnemix proxy. Each user's cache is fully isolated:

* Alice's first call misses and populates *her* cache.
* Alice's second call hits.
* Bob's first call misses — Alice's cached answer is invisible to him.
* Bob's second call hits his own cache.

Namespacing is driven by the ``X-Mnemix-Namespace`` request header
(configurable via ``CacheConfig.namespace_header``). The proxy composes
the effective namespace as ``"{tenant}::{model}"`` so separate models
also do not pollute each other, even within one tenant.

Usage:
    python examples/namespace_demo.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from mnemix.proxy import create_app
from mnemix.types import CacheConfig

MODEL = "gpt-4o-mini"
QUERY = "What's your refund policy?"


async def mock_upstream(request: httpx.Request) -> httpx.Response:
    """OpenAI stand-in: answer with a fixed support-shaped response."""
    body = json.loads(request.content)
    messages = body.get("messages", [])
    last = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    payload: dict[str, Any] = {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"(mock) answering: {last}"},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 300, "total_tokens": 400},
    }
    return httpx.Response(200, json=payload)


async def call(client: httpx.AsyncClient, *, tenant: str) -> tuple[str, float]:
    """Post the fixed query under ``tenant`` and return (cache_state, similarity)."""
    resp = await client.post(
        "/v1/chat/completions",
        headers={"X-Mnemix-Namespace": tenant},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": QUERY}],
        },
    )
    resp.raise_for_status()
    state = resp.headers.get("X-Mnemix-Cache", "UNKNOWN")
    sim = float(resp.headers.get("X-Mnemix-Similarity", "0.0"))
    return state, sim


async def run_demo() -> None:
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(mock_upstream))
    try:
        app = create_app(CacheConfig(), http_client=upstream)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://mnemix.local",
        ) as client:
            sequence: list[tuple[str, str]] = [
                ("alice", "first call  — primes alice's cache"),
                ("alice", "second call — served from alice's cache"),
                ("bob", "first call  — bob can't see alice's cache"),
                ("bob", "second call — served from bob's own cache"),
            ]

            print(f"\nquery (same for everyone): {QUERY!r}\n")
            print(f"{'tenant':<8}  {'state':<6}  {'similarity':>10}   note")
            print("-" * 74)
            for tenant, note in sequence:
                state, sim = await call(client, tenant=tenant)
                sim_str = f"{sim:.4f}" if sim > 0 else "   —   "
                print(f"{tenant:<8}  {state:<6}  {sim_str:>10}   {note}")

            metrics = (await client.get("/metrics")).json()
            total = metrics["total_requests"]
            hits = metrics["hits"]
            misses = metrics["misses"]
            print(
                f"\nproxy metrics: {total} calls, {hits} hits, {misses} misses "
                f"(expected 2 of each — one miss + one hit per tenant)",
            )
    finally:
        await upstream.aclose()


if __name__ == "__main__":
    asyncio.run(run_demo())
