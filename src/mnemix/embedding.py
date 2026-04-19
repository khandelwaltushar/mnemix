"""Embedding engines for Mnemix.

Defines the :class:`EmbeddingEngine` protocol and a local
:class:`SentenceTransformerEngine` implementation that is self-contained
(no external API calls). Model instances are cached via a factory function
so multiple callers share a single loaded model.

Example:
    Embed a single query::

        >>> import asyncio
        >>> from mnemix.embedding import get_sentence_transformer_engine
        >>> async def demo() -> int:
        ...     engine = get_sentence_transformer_engine()
        ...     vec = await engine.embed("hello world")
        ...     return len(vec)
        >>> asyncio.run(demo())  # doctest: +SKIP
        384
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class EmbeddingEngine(Protocol):
    """Protocol for async embedding providers.

    Implementations return a dense vector representation of the input text.
    The vector dimension is implementation-defined but must be stable across
    calls for a given engine instance.
    """

    async def embed(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: The input to embed. Must be non-empty for meaningful results.

        Returns:
            A list of floats of length ``self.dimension``.
        """
        ...


class SentenceTransformerEngine:
    """Local embedding engine backed by a SentenceTransformer model.

    The underlying model runs CPU-bound ``encode`` calls on a worker thread so
    coroutines aren't blocked. Construct instances via
    :func:`get_sentence_transformer_engine` to reuse a loaded model.

    Attributes:
        model_name: The SentenceTransformer model identifier.
        dimension: Output vector length.

    Example:
        >>> import asyncio
        >>> from mnemix.embedding import get_sentence_transformer_engine
        >>> async def demo() -> list[list[float]]:
        ...     engine = get_sentence_transformer_engine()
        ...     return await engine.embed_batch(["a", "b"])
        >>> asyncio.run(demo())  # doctest: +SKIP
    """

    def __init__(self, model: SentenceTransformer, model_name: str) -> None:
        """Wrap a preloaded SentenceTransformer.

        Args:
            model: A ready-to-use SentenceTransformer instance.
            model_name: The identifier used to load ``model`` (for debugging
                and metrics).
        """
        self._model = model
        self._model_name = model_name
        # sentence-transformers 5.x renamed ``get_sentence_embedding_dimension``
        # to ``get_embedding_dimension``; fall back so we support >=2.5.
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        dim = getter()
        if dim is None:
            msg = f"model {model_name!r} did not report an embedding dimension"
            raise ValueError(msg)
        self._dimension = int(dim)

    @property
    def model_name(self) -> str:
        """Return the model identifier (e.g. ``all-MiniLM-L6-v2``)."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the output vector length."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text off the event loop.

        Args:
            text: Query string to embed.

        Returns:
            Dense vector of length :attr:`dimension`.
        """
        return await asyncio.to_thread(self._encode_one, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in a single model call.

        Using a batch avoids Python-loop overhead and lets the model fuse
        matmuls. For long inputs this can be several times faster than
        calling :meth:`embed` in a loop.

        Args:
            texts: List of query strings. May be empty.

        Returns:
            A list of dense vectors, one per input, preserving order.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_many, texts)

    def _encode_one(self, text: str) -> list[float]:
        arr = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return [float(x) for x in arr]

    def _encode_many(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return [[float(x) for x in row] for row in arr]


@lru_cache(maxsize=8)
def get_sentence_transformer_engine(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformerEngine:
    """Return a memoized :class:`SentenceTransformerEngine`.

    The model is loaded on first call and reused for subsequent calls with
    the same ``model_name``. The cache lives in this function's closure —
    there is no module-level mutable state.

    Args:
        model_name: SentenceTransformer model identifier. Defaults to
            ``all-MiniLM-L6-v2`` (384 dimensions, ~22MB).

    Returns:
        The shared engine instance for ``model_name``.

    Example:
        >>> e1 = get_sentence_transformer_engine()  # doctest: +SKIP
        >>> e2 = get_sentence_transformer_engine()  # doctest: +SKIP
        >>> e1 is e2  # doctest: +SKIP
        True
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return SentenceTransformerEngine(model=model, model_name=model_name)
