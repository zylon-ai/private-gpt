import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.llm.custom.base import ZylonLLM

if TYPE_CHECKING:
    from llama_index.core.tools import BaseTool


class _FunctionCallingZylonLLM(FunctionCallingLLM, ZylonLLM):
    pass


def get_mock_function_calling_llm(
    deltas: list[list[str | ToolSelection]] | list[str | ToolSelection] | None = None,
    sleep_between_blocks: float = 0.0,
    sleep_between_deltas: float = 0.0,
) -> FunctionCallingLLM:
    if deltas is not None:
        if not deltas:
            raise ValueError("Deltas cannot be empty")

        if isinstance(deltas, list) and all(
            not isinstance(delta, list) for delta in deltas
        ):
            deltas = [deltas]

    mock_llm = MagicMock(spec=_FunctionCallingZylonLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
    mock_llm.get_metadata.return_value = mock_llm.metadata
    mock_llm.callback_manager = MagicMock()
    mock_llm.completion_to_prompt = lambda prompt, **kwargs: prompt
    mock_llm.messages_to_prompt = lambda messages, **kwargs: "\n".join(
        [message.content for message in messages or [] if message and message.content]
    )

    def get_tool_calls_from_response(
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        return tool_calls

    mock_llm.get_tool_calls_from_response = get_tool_calls_from_response

    block = 0

    async def astream_chat_with_tools(
        tools: Sequence["BaseTool"],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        nonlocal block

        if block > 0 and sleep_between_blocks > 0:
            await asyncio.sleep(sleep_between_blocks)

        for i, delta in enumerate(deltas[block]):
            if i > 0 and sleep_between_deltas > 0:
                await asyncio.sleep(sleep_between_deltas)

            message = ChatMessage(
                content=delta if isinstance(delta, str) else None,
                role=MessageRole.ASSISTANT,
                additional_kwargs={
                    "tool_calls": [delta] if isinstance(delta, ToolSelection) else None,
                },
            )
            yield ChatResponse(
                message=message,
                raw=message,
                delta=delta if isinstance(delta, str) else None,
                additional_kwargs=message.additional_kwargs,
            )

        block += 1

    async def coro(*args, **kwargs):
        return astream_chat_with_tools(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro
    return mock_llm
