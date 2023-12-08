import abc
import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt,
)

logger = logging.getLogger(__name__)


class AbstractPromptStyle(abc.ABC):
    """Abstract class for prompt styles.

    This class is used to format a series of messages into a prompt that can be
    understood by the models. A series of messages represents the interaction(s)
    between a user and an assistant. This series of messages can be considered as a
    session between a user X and an assistant Y.This session holds, through the
    messages, the state of the conversation. This session, to be understood by the
    model, needs to be formatted into a prompt (i.e. a string that the models
    can understand). Prompts can be formatted in different ways,
    depending on the model.

    The implementations of this class represent the different ways to format a
    series of messages into a prompt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("Initializing prompt_style=%s", self.__class__.__name__)

    @abc.abstractmethod
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        pass

    @abc.abstractmethod
    def _completion_to_prompt(self, completion: str) -> str:
        pass

    def messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = self._messages_to_prompt(messages)
        logger.debug("Got for messages='%s' the prompt='%s'", messages, prompt)
        return prompt

    def completion_to_prompt(self, completion: str) -> str:
        prompt = self._completion_to_prompt(completion)
        logger.debug("Got for completion='%s' the prompt='%s'", completion, prompt)
        return prompt


class DefaultPromptStyle(AbstractPromptStyle):
    """Default prompt style that uses the defaults from llama_utils.

    It basically passes None to the LLM, indicating it should use
    the default functions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Hacky way to override the functions
        # Override the functions to be None, and pass None to the LLM.
        self.messages_to_prompt = None  # type: ignore[method-assign, assignment]
        self.completion_to_prompt = None  # type: ignore[method-assign, assignment]

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return ""

    def _completion_to_prompt(self, completion: str) -> str:
        return ""


class Llama2PromptStyle(AbstractPromptStyle):
    """Simple prompt style that just uses the default llama_utils functions.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <s> [INST] <<SYS>> your system prompt here. <</SYS>>

    user message here [/INST] assistant (model) response here </s>
    ```
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return messages_to_prompt(messages)

    def _completion_to_prompt(self, completion: str) -> str:
        return completion_to_prompt(completion)


class TagPromptStyle(AbstractPromptStyle):
    """Tag prompt style (used by Vigogne) that uses the prompt style `<|ROLE|>`.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <|system|>: your system prompt here.
    <|user|>: user message here
    (possibly with context and question)
    <|assistant|>: assistant (model) response here.
    ```

    FIXME: should we add surrounding `<s>` and `</s>` tags, like in llama2?
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Format message to prompt with `<|ROLE|>: MSG` style."""
        prompt = ""
        for message in messages:
            role = message.role
            content = message.content or ""
            message_from_user = f"<|{role.lower()}|>: {content.strip()}"
            message_from_user += "\n"
            prompt += message_from_user
        # we are missing the last <|assistant|> tag that will trigger a completion
        prompt += "<|assistant|>: "
        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


def get_prompt_style(
    prompt_style: Literal["default", "llama2", "tag"] | None
) -> AbstractPromptStyle:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    if prompt_style is None or prompt_style == "default":
        return DefaultPromptStyle()
    elif prompt_style == "llama2":
        return Llama2PromptStyle()
    elif prompt_style == "tag":
        return TagPromptStyle()
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
