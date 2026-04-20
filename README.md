# Mnemix

**Stop paying for the same LLM call twice.**

Mnemix is a drop-in HTTP proxy that intercepts OpenAI and Anthropic API calls, serves semantically similar cached responses in place of upstream round-trips, and ships with configurable safety thresholds and full observability. Point your existing SDK at Mnemix by changing only `base_url` — no code changes, no monkey-patching.

[![tests](https://img.shields.io/badge/tests-209_passing-brightgreen)]() [![coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)]() [![mypy](https://img.shields.io/badge/mypy-strict-blue)]() [![python](https://img.shields.io/badge/python-3.11%2B-blue)]()

---

## The problem

Production LLM traffic is repetitive. Customer-support bots, FAQ assistants, in-product help systems, and agent loops all issue the same handful of questions over and over — just phrased slightly differently each time. A naive cache (exact-string match) catches none of it, because `"how do i reset my password?"` and `"I forgot my password — help"` hash to different keys.

At current list prices, that repetition is expensive:

| Model              | Input $/1M tok | Output $/1M tok | Blended per ~2.5K-token call |
| ------------------ | -------------: | --------------: | ---------------------------: |
| gpt-4o             |          $2.50 |          $10.00 |                    ~$0.01 |
| gpt-4-turbo        |         $10.00 |          $30.00 |                    ~$0.04 |
| claude-3-5-sonnet  |          $3.00 |          $15.00 |                    ~$0.015 |

A chatbot doing 100K calls/day on gpt-4o is ~$1,000/day. If even 40–60% of that traffic clusters around a few hundred high-frequency questions — which is typical for support and FAQ workloads — **roughly half of that bill is the same question, answered again**. Mnemix catches the rest via cosine similarity over sentence-transformer embeddings, with tenant-level isolation and TTL controls so you don't serve stale data.

---

## How it works

```
  request ──▶ ┌────────────────────────────────────────────────┐
              │ Mnemix proxy                                   │
              │                                                │
              │  1. serialise query (OpenAI or Anthropic shape)│
              │  2. bypass classifier — time-sensitive? skip   │
              │  3. embed(query)  via SentenceTransformer      │
              │  4. cosine-search within {tenant}::{model}     │
              │     ├─ score ≥ threshold ──▶ served from cache │
              │     │                         (<20 ms, no $)   │
              │     └─ miss ──▶ forward to upstream ───┐       │
              │                  store response ◀──────┘       │
              │                                                │
              │  emits: X-Mnemix-Cache: HIT|MISS               │
              │         X-Mnemix-Similarity: 0.9721            │
              └────────────────────────────────────────────────┘
```

The proxy is a FastAPI app with two endpoints that mirror upstream shapes verbatim:

- `POST /v1/chat/completions` — OpenAI chat completions
- `POST /v1/messages` — Anthropic messages

…plus `GET /metrics` for hit-rate + cost savings and `GET /health` for probes.

---

## Quickstart

### Install

```bash
pip install -e ".[dev]"
```

Mnemix is Python 3.11+. The hard deps are `pydantic`, `fastapi`, `httpx`, `redis`, `sentence-transformers`, `numpy`, and `structlog` — no secret transitive surprises.

### Option A — run the offline demo (no API key required)

The cost-savings demo mocks the upstream API so you can see the proxy end-to-end without an OpenAI or Anthropic key:

```bash
python examples/cost_savings_demo.py
```

You should see a report like:

```
============================================================
Mnemix cost-savings report — 100 calls, model=gpt-4o-mini
============================================================
Total calls:         100
Cache hits:          60 (60.0%)
Hit rate:            [########################................]  60.0%
Avg similarity:      1.0000 on hits
Tokens saved:        ~48,000
Estimated $ saved:   $0.0144 (at gpt-4o-mini published rates)
Latency:
  avg on HIT:          18.4 ms
  avg on MISS:        278.6 ms
  speedup on hit:      15.2x
============================================================
```

See also [examples/namespace_demo.py](examples/namespace_demo.py) for per-tenant isolation.

### Option B — start the live proxy against OpenAI

```bash
pip install uvicorn             # dev server; not a hard dep
export OPENAI_API_KEY=sk-...
python examples/basic_proxy.py  # listens on :8000
```

Hit it twice — the second call is served from cache:

```bash
curl -si http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-4o-mini","messages":[
        {"role":"user","content":"Summarize the plot of Hamlet"}
      ]}'
```

First response includes `X-Mnemix-Cache: MISS`. Run the same command again and you'll get `X-Mnemix-Cache: HIT` plus `X-Mnemix-Similarity: 1.0000`.

### Using from an existing OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")  # only change
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "..."}],
)
```

---

## Configuration

All knobs live on [`CacheConfig`](src/mnemix/types.py) — an immutable pydantic model you pass to `create_app(config)`:

| Field                      | Type          | Default                     | Purpose                                                            |
| -------------------------- | ------------- | --------------------------- | ------------------------------------------------------------------ |
| `similarity_threshold`     | `float`       | `0.92`                      | Minimum cosine similarity for a hit. Higher = stricter (fewer hits, lower risk of wrong answers). |
| `max_cache_size`           | `int`         | `10_000`                    | Maximum entries before LRU eviction (InMemoryStore only).          |
| `default_ttl`              | `int \| None` | `None`                      | Default TTL in seconds; `None` = no expiry.                        |
| `embedding_model`          | `str`         | `"all-MiniLM-L6-v2"`        | SentenceTransformer model name.                                    |
| `openai_base_url`          | `str`         | `"https://api.openai.com"`  | Upstream OpenAI host (override for Azure/OpenRouter/local).        |
| `anthropic_base_url`       | `str`         | `"https://api.anthropic.com"` | Upstream Anthropic host.                                         |
| `namespace_header`         | `str`         | `"X-Mnemix-Namespace"`      | Request header read to determine the per-tenant namespace.         |
| `default_namespace`        | `str`         | `"default"`                 | Namespace used when the header is absent.                          |
| `upstream_timeout_seconds` | `float`       | `60.0`                      | Connect + read timeout for forwarded upstream calls.               |

Effective cache namespace is `"{tenant}::{model}"`, so a response cached for `gpt-4o-mini` is never served to a `gpt-4-turbo` request even within one tenant. See [docs/design.md](docs/design.md) for the reasoning.

---

## Comparison vs [GPTCache](https://github.com/zilliztech/GPTCache)

GPTCache is the established prior art here and is broader in scope. We try to be honest about the trade-offs:

| Dimension                  | Mnemix                                    | GPTCache                                        |
| -------------------------- | ----------------------------------------- | ----------------------------------------------- |
| **Integration shape**      | HTTP proxy — any language, any client     | Python adapter wrapping SDK clients             |
| **Setup**                  | `pip install`, point `base_url`           | Adapter + pre/post/eval config per client       |
| **Non-Python clients**     | Works unchanged                           | Not supported                                   |
| **Storage backends**       | In-memory, Redis                          | SQLite, Redis, Milvus, Chroma, Faiss, Postgres, … |
| **Embedding providers**    | Local SentenceTransformer (pluggable protocol) | OpenAI, Cohere, HF, local, ONNX, …          |
| **Per-tenant isolation**   | Built-in via namespace header             | Manual via sessions/eval hooks                  |
| **Observability**          | Prometheus-friendly `/metrics`, HTTP headers | Limited                                       |
| **Streaming**              | Passthrough (never cached)                | Partial support                                 |
| **Bypass classifier**      | Built-in rule-based regex + TOML          | Manual pre/post processors                      |
| **Typing / code footprint**| mypy strict, Python 3.11+, ~700 LOC       | Looser typing, wider surface                    |
| **Maturity**               | Alpha (0.1)                               | Stable, widely used                             |
| **License**                | MIT                                       | MIT                                             |

**Choose GPTCache if:** you want a vector-DB smorgasbord, richer pre/post pipelines, or only need Python integration.

**Choose Mnemix if:** you want an HTTP proxy your polyglot services can share, strict typing, per-tenant metrics and namespace isolation out of the box, and a narrow surface you can fully read in an afternoon.

---

## FAQ

**Won't I get stale responses?**
Three layers guard against it:

1. **TTL per entry.** Set `default_ttl` globally or `ttl_seconds` on an entry; `SimilarityIndex` filters expired entries at read time and Redis expires them server-side.
2. **Bypass classifier.** Time-sensitive queries (`weather`, `news`, `today`, `right now`, stock prices, `currently`, …) skip the cache entirely via [`RuleBasedBypass`](src/mnemix/bypass.py). Extend the defaults via a TOML config file, or plug in your own [`BypassClassifier`](src/mnemix/bypass.py).
3. **Similarity threshold.** The 0.92 default is strict — near-identical phrasings hit, loose paraphrases do not. Raise it to 0.97 for high-stakes workloads; lower it to 0.85 for chatty assistants where a close answer is better than a fresh one.

**Does it work with Anthropic too?**
Yes. `POST /v1/messages` mirrors the Anthropic shape (including `system` as string or content-block list). The same config, same cache, same metrics.

**Can I use Redis instead of in-memory?**
```python
from mnemix import RedisStore, create_app
store = RedisStore(url="redis://localhost:6379/0")
app = create_app(store=store)
```
Redis down? Operations are wrapped in `try/except RedisError` and return cache-miss defaults, so the proxy stays up and just forwards upstream. Call `await store.ping()` at startup if you want to gate on availability.

**What about streaming responses?**
`stream=true` requests are passed through verbatim and never cached — caching a partially-consumed SSE stream is a footgun we're not willing to ship. See [docs/design.md](docs/design.md) for the reasoning and the future-work note on streaming support.

**Can I bring my own embedding model?**
Yes. [`EmbeddingEngine`](src/mnemix/embedding.py) is a Protocol. Provide any object with `async def embed(self, text: str) -> list[float]` and pass it via `create_app(embedding_engine=...)`. OpenAI `text-embedding-3-small`, Cohere, a local ONNX model — all fine. The default SentenceTransformer is chosen deliberately; see the design doc for why.

**How do I turn caching off per-request without ripping out the proxy?**
Set `stream=true` (passthrough), mark the entry `bypass=True`, or have your classifier return `True` for that query shape. The bypass path is exactly `request → upstream → response`, with a `record_miss()` so your metrics stay honest.

---

## Layout

- [src/mnemix/](src/mnemix/) — library + proxy source (strict-typed, ~700 LOC)
- [examples/](examples/) — runnable demos, no real API key needed for most
- [tests/](tests/) — pytest suite (209 tests, 93% coverage)
- [docs/design.md](docs/design.md) — architecture notes and trade-offs

## License

MIT.
