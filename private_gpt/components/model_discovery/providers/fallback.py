from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelClassificationResult,
    ModelKind,
    ModelProvider,
)
from private_gpt.components.model_discovery.providers.base import RegexModelClassifier

if TYPE_CHECKING:
    from private_gpt.components.model_discovery.models import UnclassifiedModel

logger = logging.getLogger(__name__)

ADVANCED_AUTO_DISCOVERY_DOCS = "documentation link not configured yet"  # TODO: add link


class FallbackStrategy(RegexModelClassifier):
    provider = ModelProvider.UNKNOWN

    def classify(
        self,
        unclassified: tuple[UnclassifiedModel, ...],
        *,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult:
        classified = tuple(
            ClassifiedModel(
                model=item.model,
                kind=force_kind or self.kind_from_name(item.model),
            )
            for item in unclassified
        )
        if force_kind is None:
            self._log_unknown_provider_fallback(classified)
        return ModelClassificationResult(
            provider=self.provider,
            models=classified,
        )

    def _log_unknown_provider_fallback(
        self,
        classified: tuple[ClassifiedModel, ...],
    ) -> None:
        embedding_names = [
            c.model.id for c in classified if c.kind == ModelKind.EMBEDDING
        ]
        if not embedding_names:
            return

        logger.warning(
            "Provider could not be determined. Falling back to name-based embedding "
            "detection. Models classified as embeddings: %s",
            ", ".join(sorted(embedding_names)),
        )
        logger.warning(
            "For more reliable results, use advanced auto-discovery mode. "
            "Documentation: %s",
            ADVANCED_AUTO_DISCOVERY_DOCS,
        )
