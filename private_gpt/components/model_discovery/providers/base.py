from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from private_gpt.components.model_discovery.models import (
    ClassifiedModel,
    ModelKind,
)

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput
    from private_gpt.components.model_discovery.client import DiscoveryHttpClient
    from private_gpt.components.model_discovery.models import (
        ModelClassificationResult,
        ModelProvider,
        UnclassifiedModel,
    )

EMBEDDING_MODEL_NAME_PATTERN = re.compile(
    r"(^|[-_/:.\s])("
    r"text[-_/.]?embedding"
    r"|embeddings?"
    r"|embed"
    r"|nomic[-_/.]?embed"
    r"|bge"
    r"|e5"
    r"|gte"
    r"|sentence[-_/.]?transformers?"
    r")($|[-_/:.\s])",
    re.IGNORECASE,
)


class RegexModelClassifier:
    """Shared name-based classifier used when provider metadata is not enough."""

    def classify_by_name(self, model: ModelInfoOutput) -> ClassifiedModel:
        return ClassifiedModel(model=model, kind=self.kind_from_name(model))

    def kind_from_name(self, model: ModelInfoOutput) -> ModelKind:
        text = f"{model.id} {model.display_name}"
        return (
            ModelKind.EMBEDDING
            if EMBEDDING_MODEL_NAME_PATTERN.search(text)
            else ModelKind.LLM
        )


class ModelDiscoveryStrategy(Protocol):
    """Strategy that hits its own provider-specific endpoint."""

    provider: ModelProvider

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None: ...


class OpenAICompatStrategy(Protocol):
    """Strategy that classifies pre-fetched models from /v1/models."""

    provider: ModelProvider

    def classify(
        self,
        unclassified: tuple[UnclassifiedModel, ...],
        *,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult | None: ...
