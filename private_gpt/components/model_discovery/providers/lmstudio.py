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


class LmStudioStrategy:
    provider = ModelProvider.LM_STUDIO

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None:
        classified = tuple(
            self._classify_model(item, model_info, force_kind)
            for item in self._parse_models(client.get_root_json("/api/v1/models"))
            if (model_info := model_info_from_item(self._normalize_item(item)))
            is not None
        )
        if not classified:
            return None

        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _classify_model(
        self,
        item: dict[str, Any],
        model_info: ModelInfoOutput,
        force_kind: ModelKind | None,
    ) -> ClassifiedModel:
        kind = force_kind or (
            ModelKind.EMBEDDING if item.get("type") == "embedding" else ModelKind.LLM
        )
        return ClassifiedModel(model=model_info, kind=kind)

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        if item.get("type") == "llm":
            normalized["capabilities"] = self._normalize_capabilities(
                item.get("capabilities")
            )
        return normalized

    def _parse_models(self, payload: Any | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        models = payload.get("models")
        if not isinstance(models, list):
            return []

        return [
            item
            for item in models
            if isinstance(item, dict) and item.get("type") in {"llm", "embedding"}
        ]

    def _normalize_capabilities(self, value: Any) -> dict[str, Any]:
        capabilities = value if isinstance(value, dict) else {}
        reasoning = capabilities.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        reasoning_options = reasoning.get("allowed_options")
        reasoning_options = (
            reasoning_options if isinstance(reasoning_options, list) else []
        )

        vision = capabilities.get("vision") is True
        tools = capabilities.get("trained_for_tool_use") is True
        thinking = any(
            option in {"on", "low", "medium", "high"} for option in reasoning_options
        )
        effort = {option for option in reasoning_options if isinstance(option, str)}

        supported = {"supported": True}
        unsupported = {"supported": False}

        return {
            "batch": unsupported,
            "citations": unsupported,
            "code_execution": unsupported,
            "context_management": {
                "clear_thinking_20251015": None,
                "clear_tool_uses_20250919": None,
                "compact_20260112": None,
                "supported": False,
            },
            "effort": {
                "supported": bool(effort & {"low", "medium", "high"}),
                "low": supported if "low" in effort else unsupported,
                "medium": supported if "medium" in effort else unsupported,
                "high": supported if "high" in effort else unsupported,
                "max": unsupported,
            },
            "image_input": {"supported": vision, "maximum": 1 if vision else 0},
            "audio_input": {"supported": False, "maximum": 0},
            "pdf_input": unsupported,
            "structured_outputs": {"supported": tools},
            "thinking": {
                "supported": thinking,
                "types": {
                    "adaptive": unsupported,
                    "enabled": supported if thinking else unsupported,
                },
            },
        }
