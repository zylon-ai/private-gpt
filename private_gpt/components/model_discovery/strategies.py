from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.model_discovery.providers.fallback import FallbackStrategy
from private_gpt.components.model_discovery.providers.llamacpp import LlamaCppStrategy
from private_gpt.components.model_discovery.providers.lmstudio import LmStudioStrategy
from private_gpt.components.model_discovery.providers.ollama import OllamaStrategy
from private_gpt.components.model_discovery.providers.openai import OpenAIStrategy
from private_gpt.components.model_discovery.providers.vllm import VllmStrategy

if TYPE_CHECKING:
    from private_gpt.components.model_discovery.client import DiscoveryHttpClient
    from private_gpt.components.model_discovery.models import (
        ModelClassificationResult,
        ModelKind,
    )
    from private_gpt.components.model_discovery.providers.base import (
        ModelDiscoveryStrategy,
        OpenAICompatStrategy,
    )


class StrategyChain:
    def __init__(
        self,
        discovery_strategies: tuple[ModelDiscoveryStrategy, ...] | None = None,
        openai_compat_strategies: tuple[OpenAICompatStrategy, ...] | None = None,
        fallback: FallbackStrategy | None = None,
    ) -> None:
        self._discovery_strategies = discovery_strategies or (
            OpenAIStrategy(),
            OllamaStrategy(),
            LlamaCppStrategy(),
            LmStudioStrategy(),
        )
        self._openai_compat_strategies = openai_compat_strategies or (VllmStrategy(),)
        self._fallback = fallback or FallbackStrategy()

    def discover(
        self,
        client: DiscoveryHttpClient,
        *,
        fetch_all_pages: bool,
        force_kind: ModelKind | None = None,
    ) -> ModelClassificationResult:
        # Phase 1: provider-specific endpoints
        for discover_strategy in self._discovery_strategies:
            result = discover_strategy.discover(
                client,
                fetch_all_pages=fetch_all_pages,
                force_kind=force_kind,
            )
            if result is not None:
                return result

        # Phase 2: OpenAI-compat strategies
        unclassified = client.get_unclassified_models(fetch_all_pages=fetch_all_pages)

        for classify_strategy in self._openai_compat_strategies:
            result = classify_strategy.classify(unclassified, force_kind=force_kind)
            if result is not None:
                return result

        return self._fallback.classify(unclassified, force_kind=force_kind)
