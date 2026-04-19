"""Tests for mnemix.embedding.

These tests load the real ``all-MiniLM-L6-v2`` model (≈22MB). On first run
the model is downloaded to the Hugging Face cache; subsequent runs are fast.
"""

from __future__ import annotations

import pytest

from mnemix.embedding import (
    EmbeddingEngine,
    SentenceTransformerEngine,
    get_sentence_transformer_engine,
)

MINILM_DIM = 384


@pytest.fixture(scope="module")
def engine() -> SentenceTransformerEngine:
    return get_sentence_transformer_engine()


def test_engine_conforms_to_protocol(engine: SentenceTransformerEngine) -> None:
    assert isinstance(engine, EmbeddingEngine)


def test_dimension_property(engine: SentenceTransformerEngine) -> None:
    assert engine.dimension == MINILM_DIM
    assert engine.model_name == "all-MiniLM-L6-v2"


async def test_embed_returns_correct_dimension(engine: SentenceTransformerEngine) -> None:
    vec = await engine.embed("hello world")
    assert len(vec) == MINILM_DIM
    assert all(isinstance(x, float) for x in vec)


async def test_embed_is_deterministic(engine: SentenceTransformerEngine) -> None:
    text = "the quick brown fox jumps over the lazy dog"
    v1 = await engine.embed(text)
    v2 = await engine.embed(text)
    assert v1 == v2


async def test_different_texts_produce_different_embeddings(
    engine: SentenceTransformerEngine,
) -> None:
    v1 = await engine.embed("hello")
    v2 = await engine.embed("completely different input about databases")
    assert v1 != v2


async def test_embed_batch_matches_individual_calls(
    engine: SentenceTransformerEngine,
) -> None:
    texts = ["hello", "goodbye", "the sky is blue today"]
    batch = await engine.embed_batch(texts)
    singles = [await engine.embed(t) for t in texts]

    assert len(batch) == len(texts)
    for batched_vec, single_vec in zip(batch, singles, strict=True):
        assert batched_vec == pytest.approx(single_vec, abs=1e-5)


async def test_embed_batch_preserves_order(engine: SentenceTransformerEngine) -> None:
    texts = ["alpha", "beta", "gamma"]
    batch = await engine.embed_batch(texts)
    assert batch[0] == pytest.approx(await engine.embed("alpha"), abs=1e-5)
    assert batch[2] == pytest.approx(await engine.embed("gamma"), abs=1e-5)


async def test_embed_batch_empty(engine: SentenceTransformerEngine) -> None:
    assert await engine.embed_batch([]) == []


def test_factory_caches_instances() -> None:
    e1 = get_sentence_transformer_engine()
    e2 = get_sentence_transformer_engine()
    assert e1 is e2


def test_factory_caches_per_arg_key() -> None:
    # lru_cache keys on the bound args: calling with no arg and with the
    # explicit default name are distinct cache entries. Both should work;
    # they just don't share an instance.
    e_default = get_sentence_transformer_engine()
    e_named = get_sentence_transformer_engine("all-MiniLM-L6-v2")
    assert e_default.model_name == e_named.model_name
    assert get_sentence_transformer_engine("all-MiniLM-L6-v2") is e_named
