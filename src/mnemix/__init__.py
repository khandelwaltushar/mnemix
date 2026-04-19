"""Mnemix: semantic caching layer for LLM APIs.

Mnemix intercepts OpenAI and Anthropic API calls and returns cached responses
for semantically similar queries, with configurable safety thresholds and full
observability.

Example:
    Run as a library::

        from mnemix import CacheProxy

        proxy = CacheProxy()
        # mount proxy.app into your FastAPI server

    Run as a standalone proxy::

        $ python -m mnemix.cli --port 8000
"""

from mnemix.embedding import (
    EmbeddingEngine,
    SentenceTransformerEngine,
    get_sentence_transformer_engine,
)
from mnemix.store import (
    CacheStore,
    InMemoryStore,
    SimilarityIndex,
    StoreStats,
    cosine_similarity,
)
from mnemix.types import CacheConfig, CacheEntry, CacheResult, MetricsSnapshot

__version__ = "0.1.0"

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheResult",
    "CacheStore",
    "EmbeddingEngine",
    "InMemoryStore",
    "MetricsSnapshot",
    "SentenceTransformerEngine",
    "SimilarityIndex",
    "StoreStats",
    "__version__",
    "cosine_similarity",
    "get_sentence_transformer_engine",
]
