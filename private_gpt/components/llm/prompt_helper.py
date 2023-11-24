import abc
import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (
    DEFAULT_SYSTEM_PROMPT,
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

    @abc.abstractmethod
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


class AbstractPromptStyleWithSystemPrompt(AbstractPromptStyle, abc.ABC):
    _DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(self, default_system_prompt: str | None) -> None:
        super().__init__()
        logger.debug("Got default_system_prompt='%s'", default_system_prompt)
        self.default_system_prompt = default_system_prompt


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


class Llama2PromptStyle(AbstractPromptStyleWithSystemPrompt):
    """Simple prompt style that just uses the default llama_utils functions.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <s> [INST] <<SYS>> your system prompt here. <</SYS>>

    user message here [/INST] assistant (model) response here </s>
    ```
    """

    def __init__(self, default_system_prompt: str | None = None) -> None:
        # If no system prompt is given, the default one of the implementation is used.
        super().__init__(default_system_prompt=default_system_prompt)

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return messages_to_prompt(messages, self.default_system_prompt)

    def _completion_to_prompt(self, completion: str) -> str:
        return completion_to_prompt(completion, self.default_system_prompt)


class TagPromptStyle(AbstractPromptStyleWithSystemPrompt):
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

    def __init__(self, default_system_prompt: str | None = None) -> None:
        # We have to define a default system prompt here as the LLM will not
        # use the default llama_utils functions.
        default_system_prompt = default_system_prompt or self._DEFAULT_SYSTEM_PROMPT
        super().__init__(default_system_prompt)
        self.system_prompt: str = default_system_prompt

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        messages = list(messages)
        if messages[0].role != MessageRole.SYSTEM:
            logger.info(
                "Adding system_promt='%s' to the given messages as there are none given in the session",
                self.system_prompt,
            )
            messages = [
                ChatMessage(content=self.system_prompt, role=MessageRole.SYSTEM),
                *messages,
            ]
        return self._format_messages_to_prompt(messages)

    def _completion_to_prompt(self, completion: str) -> str:
        return (
            f"<|system|>: {self.system_prompt.strip()}\n"
            f"<|user|>: {completion.strip()}\n"
            "<|assistant|>: "
        )

    @staticmethod
    def _format_messages_to_prompt(messages: list[ChatMessage]) -> str:
        """Format message to prompt with `<|ROLE|>: MSG` style."""
        assert messages[0].role == MessageRole.SYSTEM
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


def get_prompt_style(
    prompt_style: Literal["default", "llama2", "tag"] | None
) -> type[AbstractPromptStyle]:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    if prompt_style is None or prompt_style == "default":
        return DefaultPromptStyle
    elif prompt_style == "llama2":
        return Llama2PromptStyle
    elif prompt_style == "tag":
        return TagPromptStyle
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
