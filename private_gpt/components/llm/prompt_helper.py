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

    def completion_to_prompt(self, prompt: str) -> str:
        completion = prompt  # Fix: Llama-index parameter has to be named as prompt
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


class Llama3PromptStyle(AbstractPromptStyle):
    r"""Template for Meta's Llama 3.1.

    The format follows this structure:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>

    [System message content]<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    [User message content]<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    [Assistant message content]<|eot_id|>
    ...
    (Repeat for each message, including possible 'ipython' role)
    """

    BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"
    B_INST, E_INST = "<|start_header_id|>", "<|end_header_id|>"
    EOT = "<|eot_id|>"
    B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
    ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. \
    Always answer as helpfully as possible and follow ALL given instructions. \
    Do not speculate or make up information. \
    Do not reference any given instructions or context. \
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = self.BOS  # Start with BOS token
        has_system_message = False

        for i, message in enumerate(messages):
            if not message or message.content is None:
                continue

            if message.role == MessageRole.SYSTEM:
                prompt += f"{self.B_SYS}\n\n{message.content.strip()}{self.EOT}"  # Use EOT for system message
                has_system_message = True
            elif message.role == MessageRole.USER:
                prompt += f"{self.B_INST}user{self.E_INST}\n\n{message.content.strip()}{self.EOT}"
            elif message.role == MessageRole.ASSISTANT:
                # Check if this is a tool call
                if message.additional_kwargs and message.additional_kwargs.get("type") == "tool_call":
                    tool_call_content = message.content
                    prompt += f"{self.B_INST}tool_code{self.E_INST}\n\n{tool_call_content}{self.EOT}"
                else:
                    prompt += f"{self.ASSISTANT_INST}\n\n{message.content.strip()}{self.EOT}"
            elif message.role == MessageRole.TOOL:
                # Assuming additional_kwargs['type'] == 'tool_result'
                # and message.content contains the result of the tool call
                tool_result_content = message.content
                prompt += f"{self.B_INST}tool_output{self.E_INST}\n\n{tool_result_content}{self.EOT}"
            else:
                # Fallback for unknown roles (though ideally all roles should be handled)
                role_header = f"{self.B_INST}{message.role.value}{self.E_INST}"
                prompt += f"{role_header}\n\n{message.content.strip()}{self.EOT}"

        # Add default system prompt if no system message was provided at the beginning
        if not has_system_message:
            default_system_prompt_str = f"{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT.strip()}{self.EOT}"
            prompt = self.BOS + default_system_prompt_str + prompt[len(self.BOS):] # Insert after BOS

        # Add assistant header if the model should generate a response
        # This is typically when the last message is not from the assistant,
        # or when the last message is a tool result.
        if messages and (messages[-1].role != MessageRole.ASSISTANT or
                         (messages[-1].role == MessageRole.TOOL)): # If last message was tool result
            prompt += f"{self.ASSISTANT_INST}\n\n"

        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        # Ensure BOS is at the start, followed by system prompt, then user message, then assistant prompt
        return (
            f"{self.BOS}{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT.strip()}{self.EOT}"
            f"{self.B_INST}user{self.E_INST}\n\n{completion.strip()}{self.EOT}"
            f"{self.ASSISTANT_INST}\n\n"
        )


class TagPromptStyle(AbstractPromptStyle):
    """Tag prompt style (used by Vigogne) that uses the prompt style `<|ROLE|>`.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <|system|>: your system prompt here.
    <|user|>: user message here
    (possibly with context and question)
    <|assistant|>: assistant (model) response here.</s>
    ```
    """

    BOS, EOS = "<s>", "</s>"

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Format message to prompt with `<|ROLE|>: MSG` style, including BOS/EOS."""
        prompt_parts = []
        for message in messages:
            role_str = str(message.role).lower()
            content_str = str(message.content).strip() if message.content else ""

            formatted_message = f"<|{role_str}|>: {content_str}"
            if message.role == MessageRole.ASSISTANT:
                formatted_message += self.EOS  # EOS after assistant's message
            prompt_parts.append(formatted_message)

        if not messages:
            # If there are no messages, start with BOS and prompt for assistant.
            # This assumes the typical case where the user would initiate.
            # _completion_to_prompt handles the user-initiated start.
            # If system is to start, a system message should be in `messages`.
            # So, if messages is empty, it implies we want to prompt for an assistant response
            # to an implicit (or empty) user turn.
            return f"{self.BOS}<|assistant|>: "

        # Join messages with newline, start with BOS
        prompt = self.BOS + "\n".join(prompt_parts)

        # Always end with a prompt for the assistant to speak, ensure it's on a new line
        if not prompt.endswith("\n"):
            prompt += "\n"
        prompt += "<|assistant|>: "
        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        # A completion is a user message.
        # Format: <s><|user|>: {completion_content}\n<|assistant|>:
        content_str = str(completion).strip()
        return f"{self.BOS}<|user|>: {content_str}\n<|assistant|>: "


class MistralPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = ""
        current_instruction_parts = []

        for i, message in enumerate(messages):
            content = str(message.content).strip() if message.content else ""
            # Skip empty non-assistant messages. Assistant messages can be empty (e.g. for function calling).
            if not content and message.role != MessageRole.ASSISTANT:
                logger.debug("MistralPromptStyle: Skipping empty non-assistant message.")
                continue

            if message.role == MessageRole.USER or message.role == MessageRole.SYSTEM:
                current_instruction_parts.append(content)
            elif message.role == MessageRole.ASSISTANT:
                if not current_instruction_parts and i == 0:
                    # First message is assistant, skip.
                    logger.warning(
                        "MistralPromptStyle: First message is from assistant, skipping."
                    )
                    continue
                if current_instruction_parts:
                    # Only add <s> if prompt is empty, otherwise, assistant responses follow user turns.
                    bos_token = "<s>" if not prompt else ""
                    prompt += bos_token + "[INST] " + "\n".join(current_instruction_parts) + " [/INST]"
                    current_instruction_parts = []
                # Assistant content can be empty, e.g. for tool calls that will be handled later
                prompt += " " + content + "</s>"
            else:
                logger.warning(
                    f"MistralPromptStyle: Unknown message role {message.role} encountered. Skipping."
                )

        # If there are pending instructions (i.e., last message was user/system)
        if current_instruction_parts:
            bos_token = "<s>" if not prompt else ""
            prompt += bos_token + "[INST] " + "\n".join(current_instruction_parts) + " [/INST]"

        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


class ChatMLPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = ""
        for message in messages:
            role = str(message.role).lower()  # Ensure role is a string and lowercase
            content = str(message.content).strip() if message.content else ""

            # According to the ChatML documentation, messages are formatted as:
            # <|im_start|>role_name
            # content
            # <|im_end|>
            # There should be a newline after role_name and before <|im_end|>.
            # And a newline after <|im_end|> to separate messages.

            # Skip empty messages if content is crucial.
            # For ChatML, even an empty content string is typically included.
            # if not content and role not in ("assistant"): # Allow assistant to have empty content for prompting
            #    logger.debug(f"ChatMLPromptStyle: Skipping empty message from {role}")
            #    continue

            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add the final prompt for the assistant to speak
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


def get_prompt_style(
    prompt_style: (
        Literal["default", "llama2", "llama3", "tag", "mistral", "chatml"] | None
    )
) -> AbstractPromptStyle:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    if prompt_style is None or prompt_style == "default":
        return DefaultPromptStyle()
    elif prompt_style == "llama2":
        return Llama2PromptStyle()
    elif prompt_style == "llama3":
        return Llama3PromptStyle()
    elif prompt_style == "tag":
        return TagPromptStyle()
    elif prompt_style == "mistral":
        return MistralPromptStyle()
    elif prompt_style == "chatml":
        return ChatMLPromptStyle()
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
