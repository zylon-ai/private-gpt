import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from io import IOBase
from typing import Any, Protocol, runtime_checkable

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from private_gpt.components.llm.models import ReasoningEffort

logger = logging.getLogger(__name__)


class PromptData(BaseModel):
    """Data class for prompt data.

    This class is used to store the data that will be used to format a series of
    messages into a prompt. It is used to pass the data to the prompt style
    implementations.
    """

    prompt: str | None
    token_ids: list[int] | None
    # optional ids input
    images: Sequence[IOBase] | None = None
    audios: Sequence[IOBase] | None = None
    tools: Sequence[BaseTool] | None = None
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


@runtime_checkable
class MessageToPromptProtocol(Protocol):
    def __call__(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> PromptData:
        pass


@runtime_checkable
class CompletionToPromptProtocol(Protocol):
    def __call__(self, completion: str, **kwargs: Any) -> PromptData:
        pass


class PromptStyleBase(ABC):
    """Abstract class for prompt styles.

    This class is used to format a series of messages into a prompt that can be
    understood by the models. A series of messages represents the interaction(s)
    between a user and an assistant. This series of messages can be considered as a
    session between a user X and an assistant Y.This session holds, through the
    messages, the state of the conversation. This session, to be understood by the
    model, needs to be formatted into a prompt (i.e. a string that the models
    can understand). Prompts can be formatted in different ways,
    depending on the model.

    The implementations of this class represent the different ways to format
    series of messages into a prompt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("Initializing prompt_style=%s", self.__class__.__name__)

    @property
    def allow_reasoning_budget(self) -> bool:
        """Whether to allow reasoning_effort budget optimization for this prompt style.

        When enabled, the reasoning_effort budget decorator will split token allocation
        into two phases: reasoning_effort phase and completion phase. This is useful for
        models that support extended thinking/reasoning_effort.

        Returns:
            False by default. Override in subclasses to enable.
        """
        return False

    @property
    def special_tokens(self) -> list[str]:
        """Special tokens to strip from generated text for this model family.

        The text parser uses this list to build a fast regex that removes
        special tokens from model output without tokenizer round-trips.

        The default returns an empty list (nothing stripped).  Subclasses
        that have access to a tokenizer should query
        ``tokenizer.all_special_tokens``; subclasses that use a remote
        tokenizer (e.g. Triton) should hardcode the model's known special
        tokens.
        """
        return []

    @abstractmethod
    def _messages_to_prompt(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> PromptData:
        pass

    @abstractmethod
    def _completion_to_prompt(self, completion: str, **kwargs: Any) -> PromptData:
        pass

    def format_messages(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> PromptData:
        """Format messages into prompt data."""
        prompt = self._messages_to_prompt(messages, tools, reasoning_effort, **kwargs)
        logger.debug("Got for messages='%s' the prompt='%s'", messages, prompt)
        return prompt

    def format_completion(self, completion: str, **kwargs: Any) -> PromptData:
        """Format completion into prompt data."""
        prompt = self._completion_to_prompt(completion, **kwargs)
        logger.debug("Got for completion='%s' the prompt='%s'", completion, prompt)
        return prompt

    def messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str | list[int]:
        """Legacy method to format messages into a string prompt."""
        prompt = self._messages_to_prompt(messages)
        assert prompt.prompt is not None or prompt.token_ids is not None, (
            "PromptData.prompt should not be None or PromptData.token_ids should not be None"
        )
        return prompt.prompt if prompt.prompt is not None else prompt.token_ids  # type: ignore

    def completion_to_prompt(self, completion: str) -> str:
        """Legacy method to format completion into a string prompt."""
        prompt = self._completion_to_prompt(completion)
        assert prompt.prompt, "PromptData.prompt should not be None"
        return prompt.prompt

    def __call__(
        self,
        messages_or_completion: Sequence[ChatMessage] | str,
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> PromptData:
        """Unified call interface that handles both messages and completion."""
        if isinstance(messages_or_completion, str):
            return self.format_completion(messages_or_completion, **kwargs)
        else:
            return self.format_messages(
                messages_or_completion, tools, reasoning_effort, **kwargs
            )
