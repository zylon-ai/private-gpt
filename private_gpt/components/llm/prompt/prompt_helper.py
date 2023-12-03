# Ignoring the mypy check in this file, given that this file is imported only if
# running in local mode (and therefore the llama-cpp-python library is installed).
# type: ignore
"""Helper to get your llama_index messages correctly serialized into a prompt.

This set of classes and functions is used to format a series of
llama_index ChatMessage into a prompt (a unique string) that will be passed
as is to the LLM. The LLM will then use this prompt to generate a completion.

There are **MANY** formats for prompts; usually, each model has its own format.
Models posted on HuggingFace usually have a description of the format they use.
The original models, that are shipped through `transformers`, have their
format defined in the file `tokenizer_config.json` in the model's directory.
The prompt format are usually defined as a Jinja template (with some custom
Jinja token definitions). These prompt templates are usable using
the `transformers.AutoTokenizer`, as described in
https://huggingface.co/docs/transformers/main/chat_templating



Examples of `tokenizer_config.json` files:
https://huggingface.co/bofenghuang/vigogne-2-7b-chat/blob/main/tokenizer_config.json
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json
https://huggingface.co/HuggingFaceH4/zephyr-7b-beta/blob/main/tokenizer_config.json

The format of the prompt is important, as if the wrong one is used, it
will lead to "hallucinations" and other completions that are not relevant.
"""

import abc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import llama_chat_format, llama_types
from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.llama_utils import (
    DEFAULT_SYSTEM_PROMPT,
    completion_to_prompt,
    messages_to_prompt,
)

from private_gpt.constants import PROJECT_ROOT_PATH

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)


_LLAMA_CPP_PYTHON_CHAT_FORMAT: dict[str, llama_chat_format.ChatFormatter] = {
    "llama-2": llama_chat_format.format_llama2,
    "alpaca": llama_chat_format.format_alpaca,
    "vicuna": llama_chat_format.format,
    "oasst_llama": llama_chat_format.format_oasst_llama,
    "baichuan-2": llama_chat_format.format_baichuan2,
    "baichuan": llama_chat_format.format_baichuan,
    "openbuddy": llama_chat_format.format_openbuddy,
    "redpajama-incite": llama_chat_format.format_redpajama_incite,
    "snoozy": llama_chat_format.format_snoozy,
    "phind": llama_chat_format.format_phind,
    "intel": llama_chat_format.format_intel,
    "open-orca": llama_chat_format.format_open_orca,
    "mistrallite": llama_chat_format.format_mistrallite,
    "zephyr": llama_chat_format.format_zephyr,
    "chatml": llama_chat_format.format_chatml,
    "openchat": llama_chat_format.format_openchat,
}


# FIXME partial support
def llama_index_to_llama_cpp_messages(
    messages: Sequence[ChatMessage],
) -> list[llama_types.ChatCompletionRequestMessage]:
    """Convert messages from llama_index to llama_cpp format.

    Convert a list of llama_index ChatMessage to a
    list of llama_cpp ChatCompletionRequestMessage.
    """
    llama_cpp_messages: list[llama_types.ChatCompletionRequestMessage] = []
    l_msg: llama_types.ChatCompletionRequestMessage
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            l_msg = llama_types.ChatCompletionRequestSystemMessage(
                content=msg.content, role=msg.role.value
            )
        elif msg.role == MessageRole.USER:
            # FIXME partial support
            l_msg = llama_types.ChatCompletionRequestUserMessage(
                content=msg.content, role=msg.role.value
            )
        elif msg.role == MessageRole.ASSISTANT:
            # FIXME partial support
            l_msg = llama_types.ChatCompletionRequestAssistantMessage(
                content=msg.content, role=msg.role.value
            )
        elif msg.role == MessageRole.TOOL:
            # FIXME partial support
            l_msg = llama_types.ChatCompletionRequestToolMessage(
                content=msg.content, role=msg.role.value, tool_call_id=""
            )
        elif msg.role == MessageRole.FUNCTION:
            # FIXME partial support
            l_msg = llama_types.ChatCompletionRequestFunctionMessage(
                content=msg.content, role=msg.role.value, name=""
            )
        else:
            raise ValueError(f"Unknown role='{msg.role}'")
        llama_cpp_messages.append(l_msg)
    return llama_cpp_messages


def _get_llama_cpp_chat_format(name: str) -> llama_chat_format.ChatFormatter:
    logger.debug("Getting llama_cpp_python prompt_format='%s'", name)
    try:
        return _LLAMA_CPP_PYTHON_CHAT_FORMAT[name]
    except KeyError as err:
        raise ValueError(f"Unknown llama_cpp_python prompt style '{name}'") from err


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
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.nl_token = "\n"

    @abc.abstractmethod
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        pass

    @abc.abstractmethod
    def _completion_to_prompt(self, completion: str) -> str:
        pass

    def messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        logger.debug("Formatting messages='%s' to prompt", messages)
        prompt = self._messages_to_prompt(messages)
        logger.debug("Got for messages='%s' the prompt='%s'", messages, prompt)
        return prompt

    def completion_to_prompt(self, completion: str) -> str:
        logger.debug("Formatting completion='%s' to prompt", completion)
        prompt = self._completion_to_prompt(completion)
        logger.debug("Got for completion='%s' the prompt='%s'", completion, prompt)
        return prompt

    # def improve_prompt_format(self, llm: LlamaCPP) -> None:
    #     """Improve the prompt format of the given LLM.
    #
    #     Use the given metadata in the LLM to improve the prompt format.
    #     """
    #     # FIXME: we are getting IDs (1,2,13) from llama.cpp, and not actual strings
    #     llama_cpp_llm = cast(Llama, llm._model)
    #     self.bos_token = llama_cpp_llm.token_bos()
    #     self.eos_token = llama_cpp_llm.token_eos()
    #     self.nl_token = llama_cpp_llm.token_nl()
    #     print([self.bos_token, self.eos_token, self.nl_token])
    #     # (1,2,13) are the IDs of the tokens


class AbstractPromptStyleWithSystemPrompt(AbstractPromptStyle, abc.ABC):
    _DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self, default_system_prompt: str | None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        logger.debug("Got default_system_prompt='%s'", default_system_prompt)
        self.default_system_prompt = default_system_prompt

    def _add_missing_system_prompt(
        self, messages: Sequence[ChatMessage]
    ) -> Sequence[ChatMessage]:
        if messages[0].role != MessageRole.SYSTEM:
            logger.debug(
                "Adding system_promt='%s' to the given messages as there are none given in the session",
                self.default_system_prompt,
            )
            messages = [
                ChatMessage(
                    content=self.default_system_prompt, role=MessageRole.SYSTEM
                ),
                *messages,
            ]
        return messages


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
        """Dummy implementation."""
        return ""

    def _completion_to_prompt(self, completion: str) -> str:
        """Dummy implementation."""
        return ""


class LlamaIndexPromptStyle(AbstractPromptStyleWithSystemPrompt):
    """Simple prompt style that just uses the default llama_utils functions.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <s> [INST] <<SYS>> your system prompt here. <</SYS>>

    user message here [/INST] assistant (model) response here </s>
    ```
    """

    def __init__(
        self, default_system_prompt: str | None = None, *args: Any, **kwargs: Any
    ) -> None:
        # If no system prompt is given, the default one of the implementation is used.
        # default_system_prompt can be None here
        kwargs["default_system_prompt"] = default_system_prompt
        super().__init__(*args, **kwargs)

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return messages_to_prompt(messages, self.default_system_prompt)

    def _completion_to_prompt(self, completion: str) -> str:
        return completion_to_prompt(completion, self.default_system_prompt)


class VigognePromptStyle(AbstractPromptStyleWithSystemPrompt):
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

    def __init__(
        self,
        default_system_prompt: str | None = None,
        add_generation_prompt: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # We have to define a default system prompt here as the LLM will not
        # use the default llama_utils functions.
        default_system_prompt = default_system_prompt or self._DEFAULT_SYSTEM_PROMPT
        kwargs["default_system_prompt"] = default_system_prompt
        super().__init__(*args, **kwargs)
        self.add_generation_prompt = add_generation_prompt

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        messages = self._add_missing_system_prompt(messages)
        return self._format_messages_to_prompt(messages)

    def _completion_to_prompt(self, completion: str) -> str:
        messages = [ChatMessage(content=completion, role=MessageRole.USER)]
        return self._format_messages_to_prompt(messages)

    def _format_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        # TODO add BOS and EOS TOKEN !!!!! (c.f. jinja template)
        """Format message to prompt with `<|ROLE|>: MSG` style."""
        assert messages[0].role == MessageRole.SYSTEM
        prompt = ""
        # TODO enclose the interaction between self.token_bos and self.token_eos
        for message in messages:
            role = message.role
            content = message.content or ""
            message_from_user = f"<|{role.lower()}|>: {content.strip()}"
            message_from_user += self.nl_token
            prompt += message_from_user
        if self.add_generation_prompt:
            # we are missing the last <|assistant|> tag that will trigger a completion
            prompt += "<|assistant|>: "
        return prompt


class LlamaCppPromptStyle(AbstractPromptStyleWithSystemPrompt):
    def __init__(
        self,
        prompt_style: str,
        default_system_prompt: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Wrapper for llama_cpp_python defined prompt format.

        :param prompt_style:
        :param default_system_prompt: Used if no system prompt is given in the messages.
        """
        assert prompt_style.startswith("llama_cpp.")
        default_system_prompt = default_system_prompt or self._DEFAULT_SYSTEM_PROMPT
        kwargs["default_system_prompt"] = default_system_prompt
        super().__init__(*args, **kwargs)

        self.prompt_style = prompt_style[len("llama_cpp.") :]
        if self.prompt_style is None:
            return

        self._llama_cpp_formatter = _get_llama_cpp_chat_format(self.prompt_style)

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        messages = self._add_missing_system_prompt(messages)
        return self._llama_cpp_formatter(
            messages=llama_index_to_llama_cpp_messages(messages)
        ).prompt

    def _completion_to_prompt(self, completion: str) -> str:
        messages = self._add_missing_system_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )
        return self._llama_cpp_formatter(
            messages=llama_index_to_llama_cpp_messages(messages)
        ).prompt


class TemplatePromptStyle(AbstractPromptStyleWithSystemPrompt):
    def __init__(
        self,
        template_name: str,
        template_dir: str | None = None,
        add_generation_prompt: bool = True,
        default_system_prompt: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Prompt format using a Jinja template.

        :param template_name: the filename of the template to use, must be in
            the `./template/` directory.
        :param template_dir: the directory where the template is located.
            Defaults to `./template/`.
        :param default_system_prompt: Used if no system prompt is
            given in the messages.
        """
        default_system_prompt = default_system_prompt or DEFAULT_SYSTEM_PROMPT
        kwargs["default_system_prompt"] = default_system_prompt
        super().__init__(*args, **kwargs)

        self._add_generation_prompt = add_generation_prompt

        def raise_exception(message: str) -> None:
            raise TemplateError(message)

        if template_dir is None:
            self.template_dir = THIS_DIRECTORY_RELATIVE / "template"
        else:
            self.template_dir = Path(template_dir)

        self._jinja_fs_loader = FileSystemLoader(searchpath=self.template_dir)
        self._jinja_env = ImmutableSandboxedEnvironment(
            loader=self._jinja_fs_loader, trim_blocks=True, lstrip_blocks=True
        )
        self._jinja_env.globals["raise_exception"] = raise_exception

        self.template = self._jinja_env.get_template(template_name)

    @property
    def _extra_kwargs_render(self) -> dict[str, Any]:
        return {
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "nl_token": self.nl_token,
        }

    @staticmethod
    def _j_raise_exception(x: str) -> None:
        """Helper method to let Jinja template raise exceptions."""
        raise RuntimeError(x)

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        messages = self._add_missing_system_prompt(messages)
        msgs = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        return self.template.render(
            messages=msgs,
            add_generation_prompt=self._add_generation_prompt,
            **self._extra_kwargs_render,
        )

    def _completion_to_prompt(self, completion: str) -> str:
        messages = self._add_missing_system_prompt(
            [
                ChatMessage(content=completion, role=MessageRole.USER),
            ]
        )
        return self._messages_to_prompt(messages)


# TODO Maybe implement an auto-prompt style?


# Pass all the arguments at once
def get_prompt_style(
    prompt_style: str | None,
    **kwargs: Any,
) -> AbstractPromptStyle:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    if prompt_style is None:
        return DefaultPromptStyle(**kwargs)
    if prompt_style.startswith("llama_cpp."):
        return LlamaCppPromptStyle(prompt_style, **kwargs)
    elif prompt_style == "llama2":
        return LlamaIndexPromptStyle(**kwargs)
    elif prompt_style == "vigogne":
        return VigognePromptStyle(**kwargs)
    elif prompt_style == "template":
        return TemplatePromptStyle(**kwargs)
    raise ValueError(f"Unknown prompt_style='{prompt_style}'")
