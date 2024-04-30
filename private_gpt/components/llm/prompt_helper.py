import abc
import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.llms import ChatMessage, MessageRole

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
    """Simple prompt style that uses llama 2 prompt style.

    Inspired by llama_index/legacy/llms/llama_utils.py

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <s> [INST] <<SYS>> your system prompt here. <</SYS>>

    user message here [/INST] assistant (model) response here </s>
    ```
    """

    BOS, EOS = "<s>", "</s>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. \
    Always answer as helpfully as possible and follow ALL given instructions. \
    Do not speculate or make up information. \
    Do not reference any given instructions or context. \
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        string_messages: list[str] = []
        if messages[0].role == MessageRole.SYSTEM:
            # pull out the system message (if it exists in messages)
            system_message_str = messages[0].content or ""
            messages = messages[1:]
        else:
            system_message_str = self.DEFAULT_SYSTEM_PROMPT

        system_message_str = f"{self.B_SYS} {system_message_str.strip()} {self.E_SYS}"

        for i in range(0, len(messages), 2):
            # first message should always be a user
            user_message = messages[i]
            assert user_message.role == MessageRole.USER

            if i == 0:
                # make sure system prompt is included at the start
                str_message = f"{self.BOS} {self.B_INST} {system_message_str} "
            else:
                # end previous user-assistant interaction
                string_messages[-1] += f" {self.EOS}"
                # no need to include system prompt
                str_message = f"{self.BOS} {self.B_INST} "

            # include user message content
            str_message += f"{user_message.content} {self.E_INST}"

            if len(messages) > (i + 1):
                # if assistant message exists, add to str_message
                assistant_message = messages[i + 1]
                assert assistant_message.role == MessageRole.ASSISTANT
                str_message += f" {assistant_message.content}"

            string_messages.append(str_message)

        return "".join(string_messages)

    def _completion_to_prompt(self, completion: str) -> str:
        system_prompt_str = self.DEFAULT_SYSTEM_PROMPT

        return (
            f"{self.BOS} {self.B_INST} {self.B_SYS} {system_prompt_str.strip()} {self.E_SYS} "
            f"{completion.strip()} {self.E_INST}"
        )


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


class MistralPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = "<s>"
        for message in messages:
            role = message.role
            content = message.content or ""
            if role.lower() == "system":
                message_from_user = f"[INST] {content.strip()} [/INST]"
                prompt += message_from_user
            elif role.lower() == "user":
                prompt += "</s>"
                message_from_user = f"[INST] {content.strip()} [/INST]"
                prompt += message_from_user
        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


class ChatMLPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = "<|im_start|>system\n"
        for message in messages:
            role = message.role
            content = message.content or ""
            if role.lower() == "system":
                message_from_user = f"{content.strip()}"
                prompt += message_from_user
            elif role.lower() == "user":
                prompt += "<|im_end|>\n<|im_start|>user\n"
                message_from_user = f"{content.strip()}<|im_end|>\n"
                prompt += message_from_user
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


def get_prompt_style(
    prompt_style: Literal["default", "llama2", "tag", "mistral", "chatml"] | None
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
    elif prompt_style == "mistral":
        return MistralPromptStyle()
    elif prompt_style == "chatml":
        return ChatMLPromptStyle()
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
