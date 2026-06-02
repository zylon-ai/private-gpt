from __future__ import annotations

from typing import TYPE_CHECKING, Any

from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelClassificationResult,
    ModelKind,
    ModelProvider,
)

if TYPE_CHECKING:
    from private_gpt.components.model_discovery.models import UnclassifiedModel


class VllmStrategy:
    provider = ModelProvider.VLLM

    def classify(
        self,
        unclassified: tuple[UnclassifiedModel, ...],
        *,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None:
        if not unclassified:
            return None

        # vLLM signature: each model carries permission[*].allow_logprobs.
        if self._extract_allow_search_indices(unclassified[0].raw) is None:
            return None

        classified = tuple(
            ClassifiedModel(
                model=item.model,
                kind=force_kind or self._kind_from_raw(item.raw),
            )
            for item in unclassified
        )
        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _kind_from_raw(self, raw: dict[str, Any]) -> ModelKind:
        allow_search_indices = self._extract_allow_search_indices(raw)
        return ModelKind.EMBEDDING if allow_search_indices else ModelKind.LLM

    @staticmethod
    def _extract_allow_search_indices(raw: dict[str, Any]) -> bool | None:
        permissions = raw.get("permission")
        if not isinstance(permissions, list):
            return None
        for perm in permissions:
            if isinstance(perm, dict) and isinstance(
                perm.get("allow_search_indices"), bool
            ):
                return bool(perm["allow_search_indices"])
        return None
