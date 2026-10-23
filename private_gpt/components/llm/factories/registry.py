from collections.abc import Callable

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.factories.mock import MockLLMFactory
from private_gpt.components.llm.factories.openai import (
    OpenAILLMFactory,
)
from private_gpt.settings.settings import Settings

LLMProvider = type[LLMFactory] | Callable[[Settings], LLMFactory]

_PROVIDERS: dict[str, LLMProvider] = {
    "openai": OpenAILLMFactory,
    "mock": MockLLMFactory,
}


def register_llm(mode: str, provider: LLMProvider) -> None:
    _PROVIDERS[mode] = provider


class LLMFactoryRegistry:
    """Registry of LLM factories by mode."""

    def __init__(self, settings: Settings):
        self._factories: dict[str, LLMFactory] = {
            mode: provider(settings) for mode, provider in _PROVIDERS.items()
        }

    def get_factory(self, mode: str) -> LLMFactory:
        if mode not in self._factories:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"LLM mode '{mode}' is not supported. Available: {available}"
            )
        return self._factories[mode]
