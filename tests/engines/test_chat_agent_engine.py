import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat_loop.chat_loop_engine import (
    ChatLoopEngine,
    _LoopEventHandler,
    _StreamDeltaState,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.events.models import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


async def _noop_tool(value: str) -> str:
    return f"ok:{value}"


@pytest.fixture
def base_request() -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt="test"),
    )


@pytest.mark.asyncio
async def test_loop_emits_text_and_stop(base_request: ResolvedChatRequest) -> None:
    mock_llm = get_mock_function_calling_llm(["hello", " world"])
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=2,
    )

    events = [event async for event in engine.run(base_request)]
    assert any(isinstance(event, RawContentBlockDeltaEvent) for event in events)
    assert any(isinstance(event, RawMessageStopEvent) for event in events)


@pytest.mark.asyncio
async def test_loop_streams_tool_use_and_tool_result(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(
        tools=[
            ToolSpec.from_defaults(
                name="echo",
                type="echo",
                runtime="server",
                async_fn=_noop_tool,
            )
        ]
    )

    mock_llm = get_mock_function_calling_llm(
        [
            [
                ToolSelection(
                    tool_id="tool_1",
                    tool_name="echo",
                    tool_kwargs={"value": "x"},
                )
            ],
            ["done"],
        ]
    )
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=4,
    )

    events = [event async for event in engine.run(request)]
    assert any(
        isinstance(event, RawContentBlockStartEvent)
        and isinstance(event.content_block, ToolUseBlock)
        for event in events
    )
    assert any(
        isinstance(event, RawContentBlockStartEvent)
        and isinstance(event.content_block, ToolResultBlock)
        for event in events
    )


@pytest.mark.asyncio
async def test_loop_streams_reasoning_blocks(base_request: ResolvedChatRequest) -> None:
    mock_llm = MagicMock(spec=FunctionCallingLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
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
        return response.additional_kwargs.get("tool_calls", [])

    mock_llm.get_tool_calls_from_response = get_tool_calls_from_response

    async def astream_chat_with_tools(*args: Any, **kwargs: Any):
        msg_1 = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={"thinking_delta": "step-1"},
        )
        yield ChatResponse(
            message=msg_1,
            raw=msg_1,
            delta=None,
            additional_kwargs=msg_1.additional_kwargs,
        )

        msg_2 = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="done",
            additional_kwargs={"stop_reason": "end_turn"},
        )
        yield ChatResponse(
            message=msg_2,
            raw=msg_2,
            delta="done",
            additional_kwargs=msg_2.additional_kwargs,
        )

    async def coro(*args: Any, **kwargs: Any):
        return astream_chat_with_tools(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro

    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=2,
    )

    events = [event async for event in engine.run(base_request)]
    assert any(
        isinstance(event, RawContentBlockStartEvent)
        and isinstance(event.content_block, ThinkingBlock)
        for event in events
    )


@pytest.mark.asyncio
async def test_loop_accumulates_usage_across_iterations(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(
        tools=[
            ToolSpec.from_defaults(
                name="echo",
                type="echo",
                runtime="server",
                async_fn=_noop_tool,
            )
        ]
    )

    mock_llm = MagicMock(spec=FunctionCallingLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
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
        return response.additional_kwargs.get("tool_calls", [])

    mock_llm.get_tool_calls_from_response = get_tool_calls_from_response

    call_counter = 0

    async def astream_chat_with_tools(*args: Any, **kwargs: Any):
        nonlocal call_counter
        call_counter += 1

        if call_counter == 1:
            msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tool_1",
                            tool_name="echo",
                            tool_kwargs={"value": "x"},
                        )
                    ],
                    "input_tokens": 10,
                    "output_tokens": 2,
                },
            )
            yield ChatResponse(
                message=msg,
                raw=msg,
                delta=None,
                additional_kwargs=msg.additional_kwargs,
            )
            return

        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="done",
            additional_kwargs={
                "stop_reason": "end_turn",
                "input_tokens": 5,
                "output_tokens": 3,
            },
        )
        yield ChatResponse(
            message=msg,
            raw=msg,
            delta="done",
            additional_kwargs=msg.additional_kwargs,
        )

    async def coro(*args: Any, **kwargs: Any):
        return astream_chat_with_tools(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro

    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=4,
    )

    events = [event async for event in engine.run(request)]
    message_deltas = [
        event for event in events if isinstance(event, RawMessageDeltaEvent)
    ]
    assert message_deltas
    assert message_deltas[-1].usage is not None
    assert message_deltas[-1].usage.input_tokens == 15
    assert message_deltas[-1].usage.output_tokens == 5


@pytest.mark.asyncio
async def test_loop_preserves_tool_calls_when_last_chunk_has_empty_tool_calls(
    base_request: ResolvedChatRequest,
) -> None:
    request = base_request.model_copy(deep=True)
    request.tool_config = ResolvedToolConfig(
        tools=[
            ToolSpec.from_defaults(
                name="echo",
                type="echo",
                async_fn=_noop_tool,
            )
        ]
    )

    mock_llm = MagicMock(spec=FunctionCallingLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
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
        return response.additional_kwargs.get("tool_calls", [])

    mock_llm.get_tool_calls_from_response = get_tool_calls_from_response

    async def astream_chat_with_tools(*args: Any, **kwargs: Any):
        first = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id="tool_1",
                        tool_name="echo",
                        tool_kwargs={"value": "x"},
                    )
                ]
            },
        )
        yield ChatResponse(
            message=first,
            raw=first,
            delta=None,
            additional_kwargs=first.additional_kwargs,
        )

        # Provider-specific trailing chunk with empty tool_calls
        last = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            additional_kwargs={"tool_calls": []},
        )
        yield ChatResponse(
            message=last,
            raw=last,
            delta="",
            additional_kwargs=last.additional_kwargs,
        )

    async def coro(*args: Any, **kwargs: Any):
        return astream_chat_with_tools(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro

    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=2,
    )

    events = [event async for event in engine.run(request)]
    assert any(
        isinstance(event, RawContentBlockStartEvent)
        and isinstance(event.content_block, ToolUseBlock)
        for event in events
    )


@pytest.mark.asyncio
async def test_handle_stream_chunk_accumulates_token_ids_delta(
    base_request: ResolvedChatRequest,
) -> None:
    mock_llm = MagicMock(spec=FunctionCallingLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
    mock_llm.callback_manager = MagicMock()
    mock_llm.completion_to_prompt = lambda prompt, **kwargs: prompt
    mock_llm.messages_to_prompt = lambda messages, **kwargs: "\n".join(
        [message.content for message in messages or [] if message and message.content]
    )
    mock_llm.get_tool_calls_from_response = lambda *args, **kwargs: []

    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm

    engine = ChatLoopEngine(
        llm_component=llm_component,
        request_interceptors=[],
        response_interceptors=[],
        max_iterations=2,
    )

    run = engine.initialize_run(base_request)
    current_response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={},
        ),
        additional_kwargs={},
    )
    handler = _LoopEventHandler(queue=asyncio.Queue())
    stream_delta_state = _StreamDeltaState()
    lock = asyncio.Lock()

    first = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="he",
            additional_kwargs={"token_ids_delta": [11, 12]},
        ),
        delta="he",
        additional_kwargs={"token_ids_delta": [11, 12]},
    )
    second = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content="llo",
            additional_kwargs={"token_ids_delta": [13]},
        ),
        delta="llo",
        additional_kwargs={"token_ids_delta": [13]},
    )

    current_response = await engine._handle_stream_chunk(
        run=run,
        llm=mock_llm,
        chunk=first,
        current_response=current_response,
        stream_delta_state=stream_delta_state,
        handler=handler,
        tool_specs_by_name={},
        schema_by_name={},
        lock=lock,
    )
    current_response = await engine._handle_stream_chunk(
        run=run,
        llm=mock_llm,
        chunk=second,
        current_response=current_response,
        stream_delta_state=stream_delta_state,
        handler=handler,
        tool_specs_by_name={},
        schema_by_name={},
        lock=lock,
    )

    assert current_response.message.additional_kwargs["token_ids_delta"] == [11, 12, 13]
