from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelClassificationResult,
    ModelKind,
    ModelProvider,
)
from private_gpt.components.model_discovery.providers.base import RegexModelClassifier
from private_gpt.components.model_discovery.url_utils import is_openai_api_base

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput
    from private_gpt.components.model_discovery.client import DiscoveryHttpClient


OPENAI_CHAT_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)
OPENAI_NON_CHAT_MODEL_MARKERS = (
    "audio",
    "image",
    "moderation",
    "realtime",
    "sora",
    "transcribe",
    "tts",
    "whisper",
)


class OpenAIStrategy(RegexModelClassifier):
    provider = ModelProvider.OPENAI

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None:
        if not is_openai_api_base(client.api_base):
            return None

        unclassified = client.get_unclassified_models(fetch_all_pages=fetch_all_pages)
        classified = tuple(
            ClassifiedModel(
                model=item.model,
                kind=kind,
            )
            for item in unclassified
            if (kind := self._openai_model_kind(item.model, force_kind)) is not None
        )
        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _openai_model_kind(
        self,
        model: ModelInfoOutput,
        force_kind: ModelKind | None,
    ) -> ModelKind | None:
        inferred_kind = self.kind_from_name(model)
        if force_kind == ModelKind.EMBEDDING:
            return ModelKind.EMBEDDING if inferred_kind == ModelKind.EMBEDDING else None

        if inferred_kind == ModelKind.EMBEDDING:
            return inferred_kind if force_kind is None else None

        return ModelKind.LLM if self._is_openai_chat_model(model.id) else None

    def _is_openai_chat_model(self, model_id: str) -> bool:
        try:
            from llama_index.llms.openai.utils import (  # ty:ignore[unresolved-import]
                CHAT_MODELS,
                RESPONSES_API_ONLY_MODELS,
            )
        except ImportError:
            return self._looks_like_openai_chat_model(model_id)

        if model_id in RESPONSES_API_ONLY_MODELS:
            return False
        return model_id in CHAT_MODELS

    def _looks_like_openai_chat_model(self, model_id: str) -> bool:
        normalized = model_id.lower()
        if any(marker in normalized for marker in OPENAI_NON_CHAT_MODEL_MARKERS):
            return False
        return normalized.startswith(OPENAI_CHAT_MODEL_PREFIXES)
