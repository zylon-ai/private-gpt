from unittest.mock import patch

from private_gpt.components.embedding.factories.openai import OpenAIEmbeddingFactory
from private_gpt.settings.settings import (
    EmbeddingModelConfig,
    Settings,
    unsafe_settings,
)


def _settings(
    *,
    api_base: str,
    api_key: str,
    embedding_api_base: str | None,
    embedding_api_key: str | None,
) -> Settings:
    settings = Settings(**unsafe_settings)
    settings.openai.api_base = api_base
    settings.openai.api_key = api_key
    settings.openai.embedding_api_base = embedding_api_base
    settings.openai.embedding_api_key = embedding_api_key
    return settings


def _config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        name="mxbai-embed-large", mode="openai", context_window=512
    )


def test_local_openai_compatible_engine_does_not_require_api_key() -> None:
    # Regression test for #2260: a local engine (Ollama, vLLM, ...) must embed
    # without an API key instead of failing with "Missing credentials".
    settings = _settings(
        api_base="http://localhost:11434/v1",
        api_key="",
        embedding_api_base="http://localhost:11434/v1",
        embedding_api_key=None,
    )

    with patch(
        "llama_index.embeddings.openai_like.OpenAILikeEmbedding"
    ) as mock_embedding:
        OpenAIEmbeddingFactory(settings)._create_embedding(_config())

    _, kwargs = mock_embedding.call_args
    assert kwargs["api_key"]


def test_real_openai_endpoint_keeps_real_key() -> None:
    # api.openai.com must keep the real (empty) key and fail loudly instead of
    # silently receiving a placeholder key.
    settings = _settings(
        api_base="https://api.openai.com/v1",
        api_key="",
        embedding_api_base=None,
        embedding_api_key=None,
    )

    with patch("llama_index.embeddings.openai.OpenAIEmbedding") as mock_embedding:
        OpenAIEmbeddingFactory(settings)._create_embedding(_config())

    _, kwargs = mock_embedding.call_args
    assert kwargs["api_key"] == ""
