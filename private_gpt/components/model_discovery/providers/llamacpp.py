from __future__ import annotations

from typing import TYPE_CHECKING, Any

from private_gpt.components.model_discovery.client import (
    extract_model_items,
    model_info_from_item,
)
from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelClassificationResult,
    ModelProvider,
)
from private_gpt.components.model_discovery.providers.base import RegexModelClassifier

if TYPE_CHECKING:
    from private_gpt.components.model_discovery.client import DiscoveryHttpClient
    from private_gpt.components.model_discovery.models import ModelKind


class LlamaCppStrategy(RegexModelClassifier):
    provider = ModelProvider.LLAMA_CPP

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None:
        if not self._is_llamacpp(client.get_root_json("/props")):
            return None

        classified = tuple(
            ClassifiedModel(model=model_info, kind=force_kind)
            if force_kind is not None
            else self.classify_by_name(model_info)
            for item in self._merge_model_items(client.get_root_json("/models"))
            if (model_info := model_info_from_item(item)) is not None
        )
        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _is_llamacpp(self, payload: Any | None) -> bool:
        if not isinstance(payload, dict):
            return False

        return any(
            key in payload
            for key in (
                "default_generation_settings",
                "model_path",
                "chat_template",
                "chat_template_caps",
                "modalities",
                "build_info",
            )
        )

    def _merge_model_items(self, payload: Any | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        metadata_by_id = {
            item["id"]: item
            for item in extract_model_items(payload.get("data"))
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        items: list[dict[str, Any]] = []
        for item in extract_model_items(payload.get("models")):
            if not isinstance(item, dict):
                continue

            model_id = item.get("id") or item.get("model") or item.get("name")
            metadata = metadata_by_id.get(model_id, {})
            items.append({**metadata, **item})

        if items:
            return items

        return [item for item in extract_model_items(payload) if isinstance(item, dict)]
