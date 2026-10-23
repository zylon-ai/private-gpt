import abc
import enum
from collections.abc import Callable
from typing import Any

from injector import Injector
from llama_index.core.base.llms.types import ChatMessage

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.builders.summary_builder import (
    SummarizeWorkflowBuilder,
)


class BaseMemoryStrategy(abc.ABC):
    """Base class for memory strategies."""

    @abc.abstractmethod
    async def get_memory(
        self,
        chat_history: list[ChatMessage],
        max_length: int | None = None,
        **kwargs: dict[str, Any],
    ) -> list[ChatMessage]:
        """Get the memory content.

        Returns:
            str: The memory content.
        """
        pass


class CondenseStrategyType(enum.StrEnum):
    """Enum for condensing strategies."""

    CONDENSER = "condenser"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "CondenseStrategyType":
        """Convert a string to a CondenseStrategyType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


CondenseStrategyProvider = Callable[..., BaseMemoryStrategy]


def _condenser_strategy(
    injector: Injector | None = None, **kwargs: Any
) -> BaseMemoryStrategy:
    from private_gpt.components.chat.processors.chat_history.memory.strategies.condenser import (
        CondenserContextMemoryStrategy,
    )

    workflow = injector.get(SummarizeWorkflowBuilder) if injector else None
    llm_component = injector.get(LLMComponent) if injector else None
    prompt_builder_service = injector.get(PromptBuilderService) if injector else None

    return CondenserContextMemoryStrategy(
        summarize_workflow_builder=workflow,
        llm_component=llm_component,
        prompt_builder_service=prompt_builder_service,
        **kwargs,
    )


_PROVIDERS: dict[CondenseStrategyType, CondenseStrategyProvider] = {
    CondenseStrategyType.CONDENSER: _condenser_strategy,
}


def register_condense_strategy(
    strategy: CondenseStrategyType, provider: CondenseStrategyProvider
) -> None:
    _PROVIDERS[strategy] = provider


def get_condense_memory_strategy(
    strategy: str | CondenseStrategyType,
    injector: Injector | None = None,
    **kwargs: Any,
) -> BaseMemoryStrategy:
    """Get the condense strategy from a string or CondenseStrategy enum.

    Args:
        strategy (str | CondenseStrategy): The strategy to get.
        injector (Injector | None): Optional dependency injector for workflow builders.
        **kwargs: Additional keyword arguments to pass to the strategy constructor.

    Returns:
        CondenseStrategy: The corresponding CondenseStrategy enum.
    """
    strategy_enum = (
        strategy
        if isinstance(strategy, CondenseStrategyType)
        else CondenseStrategyType.from_string(strategy)
    )
    provider = _PROVIDERS.get(strategy_enum)
    if provider is None:
        raise ValueError(f"Unknown condense strategy: {strategy}")
    return provider(injector=injector, **kwargs)
