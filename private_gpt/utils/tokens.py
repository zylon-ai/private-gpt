import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import BaseTool

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.chat.processors.chat_history.memory.utils.content import (
    messages_to_history_str,
)
from private_gpt.components.llm.llm_helper import get_tokenizer
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.llm.prompt_styles.prompt_style_base import (
    MessageToPromptProtocol,
    PromptData,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
    TokenizedInput,
)

AsyncTokenizerFn = Callable[..., TokenizedInput | Awaitable[TokenizedInput]]
FullAsyncTokenizerFn = Callable[..., Awaitable[TokenizedInput]]


async def async_tokenizer(
    texts: TextLike | None = None,
    images: ImageLike | None = None,
    audios: AudioLike | None = None,
    tokenizer_fn: AsyncTokenizerFn | None = None,
) -> list[int]:
    """Tokenize text using either sync or async tokenizer function.

    Args:
        texts: Text to tokenize
        images: Optional images for multimodal tokenizers
        audios: Optional audio for multimodal tokenizers
        tokenizer_fn: Tokenizer function (sync or async)

    Returns:
        List of token IDs
    """
    if tokenizer_fn is None:
        return []

    if asyncio.iscoroutinefunction(tokenizer_fn):
        tokens: list[int] = await tokenizer_fn(texts, images, audios)
        return tokens
    else:
        result = await asyncio.to_thread(tokenizer_fn, texts, images, audios)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, TokenizedInput):
            raise TypeError("Tokenizer function must return TokenizedInput")
        return result


MessageInputProtocol = (
    MessageToPromptProtocol | Callable[[Sequence[ChatMessage]], str | list[int]]
)


async def estimate_token_count(
    chat_history: list[ChatMessage] | None,
    tools: Sequence[ToolSpec] | Sequence[BaseTool] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    tokenizer_fn: AsyncTokenizerFn | None = None,
    message_to_input: MessageInputProtocol | None = None,
    **kwargs: Any,
) -> int:
    """Estimate the number of tokens in the chat history.

    Args:
        chat_history: List of chat messages
        tools: Optional list of tools that may be included in the prompt
        reasoning_effort: Optional effort level that may affect prompt construction
        tokenizer_fn: Optional tokenizer function (sync or async)
        message_to_input: Optional function to convert messages to prompt string
        **kwargs: Additional arguments to pass to message_to_input function

    Returns:
        Estimated token count
    """
    if not chat_history:
        return 0

    tokenizer_fn = tokenizer_fn or get_tokenizer()
    messages_to_prompt = message_to_input or messages_to_history_str

    def get_tools() -> list[BaseTool] | None:
        li_tools: list[BaseTool] | None = None
        if tools is not None:
            li_tools = []
            for tool in tools:
                if isinstance(tool, BaseTool):
                    li_tools.append(tool)
                elif isinstance(tool, ToolSpec):
                    li_tools.append(tool.to_function_tool())
        return li_tools if li_tools else None

    def to_prompt_or_token_ids() -> str | list[int]:
        if isinstance(messages_to_prompt, MessageToPromptProtocol):
            sig = inspect.signature(messages_to_prompt)
            accepted = sig.parameters.keys()
            values: dict[str, Any] = {
                "tools": get_tools(),
                "reasoning_effort": reasoning_effort,
                "kwargs": kwargs,
            }
            kwargs_to_pass = {k: v for k, v in values.items() if k in accepted}

            prompt_result: PromptData = messages_to_prompt(
                chat_history, **kwargs_to_pass
            )
            if isinstance(prompt_result, PromptData):
                if prompt_result.token_ids:
                    return prompt_result.token_ids
                elif prompt_result.prompt:
                    return prompt_result.prompt
            elif isinstance(prompt_result, str) or (
                isinstance(prompt_result, list)
                and all(isinstance(i, int) for i in prompt_result)
            ):
                return prompt_result
            raise ValueError(
                "MessageToPromptProtocol must return either token_ids or prompt string."
            )
        else:
            return messages_to_prompt(chat_history)

    prompt: str | list[int] = await asyncio.to_thread(to_prompt_or_token_ids)
    if isinstance(prompt, list):
        return len(prompt)

    tokens: list[int] = await async_tokenizer(prompt, tokenizer_fn=tokenizer_fn)
    return len(tokens)
