from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from private_gpt.components.model_discovery.client import positive_int
from private_gpt.components.model_discovery.models import ModelKind
from private_gpt.components.model_discovery.service import discover_model_infos
from private_gpt.components.model_discovery.url_utils import is_openai_api_base

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput
    from private_gpt.settings.settings import EmbeddingModelConfig

logger = logging.getLogger(__name__)

DISCOVERY_MODEL_DEFAULTS: dict[str, Any] = {
    "mode": "openai",
    "enabled": True,
    "context_window": 512,
    "embedding_batch_size": 8,
    "prefix_text": None,
    "prefix_query": None,
}
DEFAULT_DISCOVERY_TIMEOUT = 3.0


def _probe_embed_dim(api_base: str, api_key: str | None, model_name: str) -> int | None:
    try:
        if is_openai_api_base(api_base):
            from llama_index.embeddings.openai import (  # ty:ignore[unresolved-import]
                OpenAIEmbedding,
            )

            embedding = OpenAIEmbedding(
                api_base=api_base, api_key=api_key, model=model_name
            )
        else:
            from llama_index.embeddings.openai_like import (  # ty:ignore[unresolved-import]
                OpenAILikeEmbedding,
            )

            embedding = OpenAILikeEmbedding(
                api_base=api_base, api_key=api_key or "no-key", model_name=model_name
            )
        test_vec = embedding.get_text_embedding("test")
        dim = len(test_vec)
        logger.info("Auto-detected embed_dim=%d for model '%s'", dim, model_name)
        return dim
    except Exception as e:
        logger.warning("Failed to probe embed_dim for '%s': %s", model_name, e)
        return None


def _model_info_to_embedding_config(
    model_info: ModelInfoOutput,
    api_base: str,
    api_key: str | None,
    mode: str | None = None,
    provider: str | None = None,
) -> EmbeddingModelConfig:
    from private_gpt.settings.settings import EmbeddingModelConfig

    defaults = DISCOVERY_MODEL_DEFAULTS
    embed_dim = positive_int(model_info.embed_dim) or _probe_embed_dim(
        api_base, api_key, model_info.id
    )
    config = {
        **defaults,
        "name": model_info.id,
        "mode": mode or str(defaults["mode"]),
        "provider": provider,
        "alias": model_info.id,
        "context_window": positive_int(model_info.max_input_tokens)
        or defaults["context_window"],
        "embed_dim": embed_dim,
    }
    return EmbeddingModelConfig(**config)


def get_embedding_models(
    api_base: str,
    api_key: str | None,
    *,
    mode: str | None = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    fetch_all_pages: bool = True,
    force_model_kind: bool = False,
) -> list[EmbeddingModelConfig]:
    discovery = discover_model_infos(
        api_base,
        api_key,
        force_kind=ModelKind.EMBEDDING if force_model_kind else None,
        timeout=timeout,
        fetch_all_pages=fetch_all_pages,
    )

    return [
        _model_info_to_embedding_config(
            model_info,
            api_base=api_base,
            api_key=api_key,
            mode=mode,
            provider=discovery.provider.value,
        )
        for model_info in discovery.embedding_models
    ]
