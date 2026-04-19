"""Simulate 100 customer-support queries and report cache savings.

No real LLM API key is required — the upstream is mocked via
``httpx.MockTransport`` so the demo runs offline. The mock sleeps 200ms
per call to simulate upstream latency; cache hits bypass that entirely,
making the latency difference visible in the report.

Traffic model: 40 unique realistic customer-support questions each asked
once, plus 60 random repeats (uniform over the 40), shuffled. With the
default similarity threshold (0.92), repeats should all hit and unique
first-occurrences should all miss — ~60% hit rate.

Usage:
    python examples/cost_savings_demo.py
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx

from mnemix.proxy import create_app
from mnemix.types import CacheConfig

UNIQUE_QUERIES: tuple[str, ...] = (
    "How do I reset my password?",
    "What's your refund policy?",
    "How long does shipping take?",
    "Can I track my order?",
    "What payment methods do you accept?",
    "Do you ship internationally?",
    "How do I cancel my subscription?",
    "Where can I find my invoice?",
    "Can I change my delivery address?",
    "What's the warranty on this product?",
    "How do I return an item?",
    "Is there a size guide?",
    "Do you offer gift wrapping?",
    "What's the status of my refund?",
    "How do I contact customer support?",
    "Can I speak to a human agent?",
    "What are your business hours?",
    "Do you have a physical store?",
    "How do I apply a discount code?",
    "Can I combine multiple discount codes?",
    "Is my payment information secure?",
    "Why was my card declined?",
    "How do I update my billing information?",
    "Can I pay in installments?",
    "Do you offer bulk discounts?",
    "How do I create an account?",
    "I forgot my username, what do I do?",
    "Why didn't I receive a confirmation email?",
    "Can I change the email on my account?",
    "How do I unsubscribe from marketing emails?",
    "Is my personal data safe with you?",
    "What's in your privacy policy?",
    "Do you sell my data to third parties?",
    "How can I delete my account?",
    "Why is my order delayed?",
    "Can I upgrade to expedited shipping?",
    "What happens if my package is lost?",
    "Do you offer installation services?",
    "How do I request a replacement item?",
    "Is there a loyalty or rewards program?",
)

MODEL = "gpt-4o-mini"
UPSTREAM_LATENCY_S = 0.2  # simulated upstream round-trip per miss
PROMPT_TOKENS = 220
COMPLETION_TOKENS = 580  # realistic support-answer length
TOTAL_TOKENS = PROMPT_TOKENS + COMPLETION_TOKENS

TOTAL_CALLS = 100
UNIQUE_COUNT = len(UNIQUE_QUERIES)
REPEAT_COUNT = TOTAL_CALLS - UNIQUE_COUNT


def build_traffic(seed: int = 42) -> list[str]:
    """Produce a 100-item query sequence: 40 unique + 60 random repeats."""
    rng = random.Random(seed)
    repeats = [rng.choice(UNIQUE_QUERIES) for _ in range(REPEAT_COUNT)]
    sequence = [*UNIQUE_QUERIES, *repeats]
    rng.shuffle(sequence)
    return sequence


async def mock_upstream(request: httpx.Request) -> httpx.Response:
    """Pretend to be OpenAI: sleep briefly, echo back a support-shaped reply."""
    await asyncio.sleep(UPSTREAM_LATENCY_S)
    body = json.loads(request.content)
    messages = body.get("messages", [])
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    response: dict[str, Any] = {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"(mock reply to: {last_user[:40]}...)",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": TOTAL_TOKENS,
        },
    }
    return httpx.Response(200, json=response)


async def run_demo() -> None:
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(mock_upstream))
    try:
        app = create_app(CacheConfig(), http_client=upstream)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://mnemix.local",
        ) as client:
            sequence = build_traffic()
            hit_latencies_ms: list[float] = []
            miss_latencies_ms: list[float] = []

            print(f"running {TOTAL_CALLS} calls against the Mnemix proxy...\n")
            for i, query in enumerate(sequence, start=1):
                start = time.perf_counter()
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": query}],
                    },
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                marker = resp.headers.get("X-Mnemix-Cache", "UNKNOWN")
                if marker == "HIT":
                    hit_latencies_ms.append(elapsed_ms)
                else:
                    miss_latencies_ms.append(elapsed_ms)
                if i % 20 == 0:
                    print(f"  {i:3d}/{TOTAL_CALLS} done")

            metrics = (await client.get("/metrics")).json()

        _print_report(metrics, hit_latencies_ms, miss_latencies_ms)
    finally:
        await upstream.aclose()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _print_report(
    metrics: dict[str, Any],
    hit_ms: list[float],
    miss_ms: list[float],
) -> None:
    total = int(metrics["total_requests"])
    hits = int(metrics["hits"])
    misses = int(metrics["misses"])
    hit_rate = float(metrics["hit_rate"])
    tokens = int(metrics["estimated_tokens_saved"])
    cost = float(metrics["estimated_cost_saved_usd"])
    avg_sim = float(metrics["avg_similarity_on_hit"])

    hit_avg = _mean(hit_ms)
    miss_avg = _mean(miss_ms)
    speedup = (miss_avg / hit_avg) if hit_avg > 0 else float("inf")

    bar_width = 40
    filled = round(hit_rate * bar_width)
    bar = "#" * filled + "." * (bar_width - filled)

    print(
        f"""
{"=" * 60}
Mnemix cost-savings report — {total} calls, model={MODEL}
{"=" * 60}
Total calls:         {total}
Cache hits:          {hits} ({hit_rate * 100:.1f}%)
Cache misses:        {misses}
Hit rate:            [{bar}]  {hit_rate * 100:.1f}%

Avg similarity:      {avg_sim:.4f} on hits
Tokens saved:        ~{tokens:,}
Estimated $ saved:   ${cost:.4f} (at {MODEL} published rates)

Latency:
  avg on HIT:        {hit_avg:6.1f} ms
  avg on MISS:       {miss_avg:6.1f} ms
  speedup on hit:    {speedup:6.1f}x
  (hits skip the {UPSTREAM_LATENCY_S * 1000:.0f}ms upstream round-trip)
{"=" * 60}
""",
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
