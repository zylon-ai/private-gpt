from __future__ import annotations

from typing import TYPE_CHECKING, Any

from private_gpt.components.model_discovery.client import model_info_from_item
from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelClassificationResult,
    ModelKind,
    ModelProvider,
)

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput
    from private_gpt.components.model_discovery.client import DiscoveryHttpClient

OLLAMA_EMBEDDING_FAMILIES = {
    "bert",
    "nomic-bert",
    "sentence-transformers",
}


class OllamaStrategy:
    provider = ModelProvider.OLLAMA

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None:
        classified = tuple(
            self._classify_tag(tag, model_info, force_kind)
            for tag in self._parse_tags(client.get_root_json("/api/tags"))
            if (model_info := model_info_from_item(tag)) is not None
        )
        if not classified:
            return None

        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _classify_tag(
        self,
        tag: dict[str, Any],
        model_info: ModelInfoOutput,
        force_kind: ModelKind | None,
    ) -> ClassifiedModel:
        kind = force_kind or (
            ModelKind.EMBEDDING if self._tag_indicates_embedding(tag) else ModelKind.LLM
        )
        return ClassifiedModel(model=model_info, kind=kind)

    def _parse_tags(self, payload: Any | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        models = payload.get("models")
        if not isinstance(models, list):
            return []

        return [item for item in models if isinstance(item, dict)]

    def _tag_indicates_embedding(self, tag: dict[str, Any]) -> bool:
        details = tag.get("details")
        details = details if isinstance(details, dict) else {}

        family = details.get("family")
        families = details.get("families")
        capabilities = tag.get("capabilities")
        model_type = tag.get("type")

        return (
            (isinstance(family, str) and family.lower() in OLLAMA_EMBEDDING_FAMILIES)
            or (
                isinstance(families, list)
                and any(
                    isinstance(item, str) and item.lower() in OLLAMA_EMBEDDING_FAMILIES
                    for item in families
                )
            )
            or (
                isinstance(capabilities, list)
                and any(
                    isinstance(item, str) and item.lower() in {"embed", "embedding"}
                    for item in capabilities
                )
            )
            or (isinstance(model_type, str) and model_type.lower() == "embedding")
        )
