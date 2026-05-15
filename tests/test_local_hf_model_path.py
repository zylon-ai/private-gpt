"""Tests for the HuggingFace local model path feature (issue #1625)."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_settings(
    *,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    model_path: str | None = None,
    access_token: str | None = None,
    trust_remote_code: bool = False,
) -> MagicMock:
    """Build a minimal mock Settings object for the embedding component."""
    hf = MagicMock()
    hf.embedding_hf_model_name = model_name
    hf.embedding_hf_model_path = model_path
    hf.access_token = access_token
    hf.trust_remote_code = trust_remote_code

    embedding = MagicMock()
    embedding.mode = "huggingface"

    settings = MagicMock()
    settings.huggingface = hf
    settings.embedding = embedding
    return settings


@patch(
    "private_gpt.components.embedding.embedding_component.models_cache_path",
    "/tmp/test_cache",
)
@patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
def test_uses_hub_name_when_no_local_path(mock_hf_embed: MagicMock) -> None:
    """Default behaviour: model is loaded by HuggingFace Hub name."""
    from private_gpt.components.embedding.embedding_component import EmbeddingComponent

    settings = _make_settings(model_name="BAAI/bge-small-en-v1.5")

    with patch.object(EmbeddingComponent, "__init__", lambda self, s: None):
        comp = EmbeddingComponent.__new__(EmbeddingComponent)

    comp.__class__.__init__(comp, settings)

    mock_hf_embed.assert_called_once()
    call_kwargs = mock_hf_embed.call_args
    assert call_kwargs.kwargs["model_name"] == "BAAI/bge-small-en-v1.5"


@patch(
    "private_gpt.components.embedding.embedding_component.models_cache_path",
    "/tmp/test_cache",
)
@patch(
    "private_gpt.components.embedding.embedding_component.absolute_or_from_project_root",
)
@patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
def test_uses_local_path_when_configured(
    mock_hf_embed: MagicMock,
    mock_resolve: MagicMock,
) -> None:
    """When embedding_hf_model_path is set, it takes precedence over model_name."""
    from private_gpt.components.embedding.embedding_component import EmbeddingComponent

    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = True
    fake_path.__str__ = lambda self: "/models/my-custom-embedding"
    mock_resolve.return_value = fake_path

    settings = _make_settings(
        model_name="BAAI/bge-small-en-v1.5",
        model_path="/models/my-custom-embedding",
    )

    with patch.object(EmbeddingComponent, "__init__", lambda self, s: None):
        comp = EmbeddingComponent.__new__(EmbeddingComponent)

    comp.__class__.__init__(comp, settings)

    mock_hf_embed.assert_called_once()
    call_kwargs = mock_hf_embed.call_args
    assert call_kwargs.kwargs["model_name"] == "/models/my-custom-embedding"


@patch(
    "private_gpt.components.embedding.embedding_component.models_cache_path",
    "/tmp/test_cache",
)
@patch(
    "private_gpt.components.embedding.embedding_component.absolute_or_from_project_root",
)
@patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding")
def test_raises_when_local_path_does_not_exist(
    mock_hf_embed: MagicMock,
    mock_resolve: MagicMock,
) -> None:
    """ValueError should be raised if the configured local path doesn't exist."""
    import pytest

    from private_gpt.components.embedding.embedding_component import EmbeddingComponent

    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = False
    mock_resolve.return_value = fake_path

    settings = _make_settings(model_path="/nonexistent/path")

    with patch.object(EmbeddingComponent, "__init__", lambda self, s: None):
        comp = EmbeddingComponent.__new__(EmbeddingComponent)

    with pytest.raises(ValueError, match="does not exist"):
        comp.__class__.__init__(comp, settings)


def test_settings_model_accepts_local_path() -> None:
    """HuggingFaceSettings schema accepts the new field."""
    from private_gpt.settings.settings import HuggingFaceSettings

    s = HuggingFaceSettings(
        embedding_hf_model_name="test-model",
        embedding_hf_model_path="/opt/models/test-model",
    )
    assert s.embedding_hf_model_path == "/opt/models/test-model"


def test_settings_model_local_path_defaults_to_none() -> None:
    """Local path should be None by default."""
    from private_gpt.settings.settings import HuggingFaceSettings

    s = HuggingFaceSettings(embedding_hf_model_name="test-model")
    assert s.embedding_hf_model_path is None
