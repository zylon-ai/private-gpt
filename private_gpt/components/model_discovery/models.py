from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from private_gpt.chat.input_models import ModelInfoOutput


class ModelProvider(StrEnum):
    OPENAI = "openai"
    LLAMA_CPP = "llamacpp"
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    VLLM = "vllm"
    UNKNOWN = "unknown"


class ModelKind(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class UnclassifiedModel:
    model: ModelInfoOutput
    raw: dict[str, Any]


@dataclass(frozen=True)
class ClassifiedModel:
    model: ModelInfoOutput
    kind: ModelKind


@dataclass(frozen=True)
class ModelClassificationResult:
    provider: ModelProvider
    models: tuple[ClassifiedModel, ...]

    @property
    def llm_models(self) -> list[ModelInfoOutput]:
        return [
            classified.model
            for classified in self.models
            if classified.kind == ModelKind.LLM
        ]

    @property
    def embedding_models(self) -> list[ModelInfoOutput]:
        return [
            classified.model
            for classified in self.models
            if classified.kind == ModelKind.EMBEDDING
        ]


@dataclass(frozen=True)
class ModelDiscoveryResult:
    provider: ModelProvider
    models: tuple[ModelInfoOutput, ...]
    llm_models: tuple[ModelInfoOutput, ...]
    embedding_models: tuple[ModelInfoOutput, ...]

    @classmethod
    def from_classified(
        cls,
        provider: ModelProvider,
        classified: tuple[ClassifiedModel, ...],
    ) -> ModelDiscoveryResult:
        return cls(
            provider=provider,
            models=tuple(item.model for item in classified),
            llm_models=tuple(
                item.model for item in classified if item.kind == ModelKind.LLM
            ),
            embedding_models=tuple(
                item.model for item in classified if item.kind == ModelKind.EMBEDDING
            ),
        )
