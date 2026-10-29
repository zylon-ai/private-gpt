"""Regression tests: the per-model ``embedding_batch_size`` must reach the model.

llama_index splits provider requests by ``BaseEmbedding.embed_batch_size``:
``get_text_embedding_batch`` flushes its pending texts every time the batch
reaches that size. The embedding factories never passed the configured value,
so every batch sent to the embedding API kept the library default (100 texts
for OpenAI, 10 for the rest) and ``embedding_batch_size`` in the model config
was silently ignored.
"""

import pytest
from llama_index.core import MockEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

from private_gpt.components.embedding.factories.mock import MockEmbeddingFactory
from private_gpt.components.embedding.factories.openai import OpenAIEmbeddingFactory
from private_gpt.settings.settings import (
    EmbeddingModelConfig,
    Settings,
    unsafe_settings,
)


def _config(batch_size: int | None) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        name="batching-model",
        mode="mock",
        context_window=512,
        embedding_batch_size=batch_size,
    )


def _factory() -> MockEmbeddingFactory:
    return MockEmbeddingFactory(Settings(**unsafe_settings))


def test_configured_batch_size_reaches_the_embedding_model() -> None:
    embedding = _factory().create_embedding(_config(3)).embedding

    assert embedding.embed_batch_size == 3


def test_unset_batch_size_keeps_the_library_default() -> None:
    embedding = _factory().create_embedding(_config(None)).embedding

    assert embedding.embed_batch_size == MockEmbedding(512).embed_batch_size


def test_configured_batch_size_is_the_size_of_each_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = BaseEmbedding._get_text_embeddings

    def recording(self: BaseEmbedding, texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(MockEmbedding, "_get_text_embeddings", recording)

    embedding = _factory().create_embedding(_config(3)).embedding
    embedding.get_text_embedding_batch(["a", "b", "c", "d", "e"])

    assert calls == [3, 2]


def test_configured_batch_size_reaches_the_openai_embedding_model() -> None:
    settings = Settings(**unsafe_settings)
    settings.openai.api_base = "https://api.openai.com/v1"
    settings.openai.api_key = "test-key"
    settings.openai.embedding_api_base = None
    settings.openai.embedding_api_key = None

    embedding = (
        OpenAIEmbeddingFactory(settings)
        .create_embedding(
            EmbeddingModelConfig(
                name="text-embedding-3-small",
                mode="openai",
                context_window=512,
                embedding_batch_size=7,
            )
        )
        .embedding
    )

    assert embedding.embed_batch_size == 7
