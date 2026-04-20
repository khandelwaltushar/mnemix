# Mnemix design notes

This document records the non-obvious design decisions baked into Mnemix — the "why" behind things that look arbitrary in the code. It's a companion to the [README](../README.md), not a substitute for reading the source; the source is short.

---

## Why cosine similarity, not dot product

Dot product of embeddings `a · b` equals `cos(θ) × |a| × |b|`. The magnitude terms leak into the ranking unless the vectors are unit-normalised.

SentenceTransformer outputs are **not** unit-normalised by default. Magnitudes drift with text length and token choice — longer queries get slightly larger embeddings, which would systematically make them "look more similar" under plain dot product. That's a ranking bias we don't want in a cache.

Cosine strips the magnitude: `cos(θ) = (a · b) / (|a| |b|)`. What's left is just the angle between the vectors, which is the only thing `"how close in meaning are these two texts?"` should depend on.

**Why not normalise-then-dot?** That's mathematically identical to cosine but splits the normalisation across write time (the store) and read time (the query). You'd have to remember to normalise every `query_embedding` on both sides, and any future code that forgets silently corrupts the index. Cosine-at-search-time is one place where the invariant lives — harder to break.

**What about MIPS/IP vector indexes (FAISS, Pinecone, Redis VSS)?** They usually expose both a cosine and an IP distance metric. When we upgrade to a native vector index (see limitations), we'll pick cosine there too. If the backend lets us also store unit-normalised vectors server-side, dot becomes cosine for free — but we'll enforce that in the writer, not leave it to chance.

Cosine lives in `[-1.0, 1.0]`; we clamp to that range to absorb float32 round-off (identical float32 vectors can yield `1.0 + ε`, which would fail pydantic's `le=1.0` validator on `CacheResult.similarity_score`). See [`cosine_similarity`](../src/mnemix/store.py).

---

## Why local embeddings (all-MiniLM-L6-v2), not OpenAI `text-embedding-ada-002`

The default [`EmbeddingEngine`](../src/mnemix/embedding.py) is a local SentenceTransformer (`all-MiniLM-L6-v2`, 384-dim). We could have used ada-002 or `text-embedding-3-small`; we chose not to. Four reasons:

### Latency

Every cache lookup embeds the query. An ada-002 call is a real HTTP request — 100–300 ms from most regions. A local all-MiniLM embed is 5–10 ms for a short prompt.

The cache-hit target is "sub-20 ms end-to-end". If the embedding step alone takes 300 ms, we've already lost. On a miss, the 300 ms adds straight to upstream latency. There's no world where paying a network round-trip to *decide whether to avoid a network round-trip* is the right choice.

### Cost

ada-002 is cheap — about $0.0001 per 1K tokens. Cheap is not free. A service doing 1M cache lookups/day on 500-token prompts pays ~$50/day just to embed. That's a small fraction of the LLM savings, but it's also a small fraction of nothing: local embeddings are literally $0 per call.

### Privacy

Every ada-002 call sends the query to OpenAI. For many users, **reducing upstream calls is part of why they want a cache** — compliance, data-residency, or just "we don't want our prompts leaving the VPC". Routing every cache lookup through OpenAI defeats that entirely.

Local embeddings stay on the box. Audit boundary = process boundary.

### Failure mode

ada-002 depends on OpenAI. If OpenAI is down, the embedder is down, and the cache cannot even answer "is this a hit?" — it has to either error or bypass. Local embedders don't fail independently of your own process.

### Quality trade-off, honestly stated

ada-002 is a stronger model than all-MiniLM in absolute terms (larger, more training data, higher MTEB scores on hard retrieval tasks). For **cache-hit detection of common paraphrases**, that gap is small: all-MiniLM easily distinguishes "how do I reset my password" from "how do I reset my 2FA" while scoring close paraphrases near 1.0.

If you need ada-002 or a Cohere model or your own fine-tuned embedder, [`EmbeddingEngine`](../src/mnemix/embedding.py) is a one-method Protocol. Provide any object with `async def embed(self, text: str) -> list[float]` and pass it via `create_app(embedding_engine=...)`. The default is not a constraint — it's a safe default.

---

## The threshold calibration problem

`similarity_threshold` (default `0.92`) is the single most consequential tuning knob. There is no universally correct value. It's a precision/recall trade-off over *your* traffic, and it needs to be calibrated per workload.

### The trade-off

- **Too high** (e.g. `0.99`): precision ↑, recall ↓. You only hit on near-identical queries; most paraphrases miss. The cache works, but doesn't save you much.
- **Too low** (e.g. `0.80`): recall ↑, precision ↓. You catch broad paraphrases, *and* you occasionally serve the wrong answer — "how do I cancel my account" and "how do I cancel my order" can be 0.85+ on general-purpose embeddings.

A wrong cache hit is worse than a miss. Missing costs tokens; a wrong hit costs user trust. When in doubt, err high.

### How to pick a number

1. **Collect a labelled pair set from your real traffic.** 500 pairs is enough. Each pair: `(query_a, query_b, should_hit)`. "Should hit" means "serving A's response to B is acceptable for your product".
2. **Embed every query and compute cosine for every pair.** Plot the distribution of cosines for `should_hit=True` vs `should_hit=False`.
3. **Pick the threshold** at the minimum precision you're willing to ship — usually ≥99% for anything user-facing. For support chatbots, 0.92 is a common landing spot. For medical / legal / financial Q&A, 0.97+.
4. **Shadow-mode first.** Run the cache in log-only mode (record what it *would* have served, don't actually serve it) for a week. Spot-check the would-be hits. Only flip it live when you're comfortable.
5. **Re-calibrate on model change.** Swapping embedders invalidates the threshold entirely — cosine scores don't transfer between models.

### Signals you can watch

[`MetricsSnapshot`](../src/mnemix/types.py) emits `avg_similarity_on_hit`. If that number sits at 0.98+ in production, you're leaving hit-rate on the table — most of your hits are near-duplicates and you could lower the threshold without admitting materially fuzzier matches. If it sits right at the threshold, you're in the danger zone and should raise it or tighten the bypass classifier.

The threshold is also orthogonal to the bypass classifier. Bypass handles "this category of query should *never* be cached regardless of similarity" (weather, news, stock prices). Threshold handles "how close is close enough within the cacheable subset". Use both.

---

## Why a proxy, not an SDK wrapper

The obvious alternative design is a client-library adapter — something you import in Python and wrap around `OpenAI()`. That's how GPTCache and most prior-art tools work. We deliberately chose the proxy shape.

**Language-agnostic.** Your Go backend, your Ruby worker, and your Python agent all benefit from one Mnemix instance. An SDK wrapper requires a port for every language you ship in — and you have to keep all ports in sync when the upstream API changes.

**No monkey-patching.** SDK wrappers either subclass provider clients (fragile when clients change) or monkey-patch global state (worse). A proxy doesn't touch the SDK at all — you change `base_url` and the SDK has no idea Mnemix is there. When OpenAI ships a breaking SDK change, Mnemix keeps working.

**Centralised observability.** Every request goes through one place. `/metrics` is a single source of truth for hit rate, token savings, and latency. With SDK wrappers, each service collects its own stats — and the one service that forgot to wire up caching is invisible until the bill arrives.

**Shared cache across services.** Two microservices asking the same question pay once. With SDK wrappers, each process has its own cache (unless you explicitly set up a shared backend, which is more config to get wrong).

**Uniform handling of streaming, auth, and edge cases.** The proxy forwards headers (`Authorization`, etc.) verbatim. Streaming responses pass through at the HTTP layer. `5xx` from upstream is passed back as-is and explicitly not cached. These behaviours are implemented once, at the protocol boundary, rather than once per language binding.

**Cost of the choice.** A proxy adds one extra network hop. In practice that's <5 ms inside a cluster or on-host, which rounds to zero next to the 100–500 ms of a real LLM call. On a cache hit (which skips the upstream entirely), the proxy is still strictly faster than any uncached path. Nothing we've measured makes us regret the choice.

---

## Known limitations and future work

Mnemix is 0.1. What's shipped works and is tested, but there are real edges.

### Similarity search is O(N) per query

[`SimilarityIndex`](../src/mnemix/store.py) iterates every entry in a namespace and computes cosine against each. That's fine at 10K entries per namespace; it starts to bite at 100K+. The obvious upgrade is a native vector index.

**Planned:** a Redis Stack / Redis VSS backend that uses HNSW (or FLAT for small caches) server-side, so similarity search becomes a single KNN query instead of N round-trips. [`RedisStore`](../src/mnemix/redis_store.py)'s module docstring contains the full upgrade path, including the required storage change (embedding bytes instead of JSON) and the `FT.CREATE` schema. Not default because Redis Stack is a heavier operational dependency than stock Redis.

### Streaming responses are never cached

`stream=true` requests are passed through verbatim. The cached response path ([`proxy._handle`](../src/mnemix/proxy.py)) only sees the streaming branch and records a miss. We do this because:

- Caching a stream means buffering it fully before returning — which defeats streaming for every miss.
- Replaying a cached stream as a real stream is possible but requires invented chunk boundaries and fake-realtime pacing, which isn't free.

**Planned:** an optional `cache_streams` mode with two policies — `buffer` (miss = buffer-then-return, subsequent hits stream instantly) and `tee` (stream to client while simultaneously buffering for storage). The right default is unclear; probably `off` will remain the default.

### No multi-modal support

Embeddings are text-only. Image inputs in OpenAI and Anthropic requests are serialised as plain text descriptions (image URLs etc.) and embedded through the same SentenceTransformer, which captures almost nothing useful about the image.

**Planned:** a multi-modal `EmbeddingEngine` (OpenCLIP or similar) and a content-type-aware serialiser that splits text and image parts, embeds them separately, and concatenates or attention-pools the vectors.

### No per-cost cap on queries

Every cacheable query pays the same embedding cost, regardless of its upstream token cost. A 50-token prompt costing $0.0001 upstream is treated the same as a 20K-token prompt costing $0.50. Production users probably want a budget: "don't bother caching anything below $X in projected savings". The metrics surface has enough signal to build this as a pre-filter, but it isn't shipped.

### LLM-based bypass classifier

The architecture accounts for an LLM-backed [`BypassClassifier`](../src/mnemix/bypass.py) ("is this query time-sensitive? is it personal? is it a one-shot code review?"), but only the rule-based default is shipped. Adding an LLM-backed classifier trades embedding cost for bypass-decision cost — interesting for high-stakes deployments where the rule-based regex is too blunt.

### CLI

`pyproject.toml` advertises a `mnemix` console script (`[project.scripts] mnemix = "mnemix.cli:main"`), but the CLI module isn't implemented yet. For now, use `python examples/basic_proxy.py` or mount `create_app()` into your own ASGI server.

### Authentication on management endpoints

`/metrics` and `/health` are currently unauthenticated. For production deployments, put Mnemix behind a reverse proxy (nginx, Envoy, Traefik) that handles auth on those paths. We won't add auth directly to Mnemix — it's out of scope for a cache proxy and every team has different opinions about how to do it.

### Cache invalidation is TTL + namespace clear only

No tag-based or pattern-based invalidation. If you ship a new version of your prompt template and want to wipe everything cached under it, the paths are: wait out TTL, call `store.clear(namespace)`, or roll the namespace (include a prompt-template version in the namespace composition). The last is the cleanest operationally.
