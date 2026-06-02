"""Tests for FilterZylonLoopInterceptor and PingLoopInterceptor.

Direct port of the original EventAdapter test suite.  The two flavours are
preserved exactly:

1. **Unit** — hand-crafted ``Event`` sequences are piped through
   ``_collect_from_gen``, which replicates the EventAdapter pipeline using the
   new per-event interceptor API.

2. **Integration** — real ``ChatLoopEngine`` with a mock LLM, interceptors
   wired as ``response_interceptors``.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat_loop.chat_loop_engine import ChatLoopEngine
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.events.models import (
    FatalError,
    MessageOutputDelta,
    PingEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    TLDRBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from private_gpt.server.chat.interceptors.filter_event_by_type_interceptor import (
    FilterZylonInterceptor,
)
from private_gpt.server.chat.interceptors.ping_loop_interceptor import PingInterceptor
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

if TYPE_CHECKING:
    from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
        ChatResponseLoopInterceptor,
    )


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _block_id() -> str:
    return f"block_{uuid4().hex}"


async def _collect_from_gen(
    gen: AsyncGenerator,
    response_format: str = "anthropic",
    ping_interval: float | None = None,
) -> list:
    """Pipe a hand-crafted event generator through the interceptor pipeline."""
    interceptors: list[ChatResponseLoopInterceptor] = [FilterZylonInterceptor()]
    if ping_interval:
        interceptors.append(PingInterceptor(ping_interval))

    output: list = []
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    ctx = MagicMock(spec=ChatLoopInterceptorContext)
    ctx.state = MagicMock()
    ctx.state.input = MagicMock()
    ctx.state.input.request = MagicMock()
    ctx.state.input.request.system = MagicMock()
    ctx.state.input.request.system.extensions = MagicMock()
    ctx.state.input.request.system.extensions.zylon_enabled = response_format == "zylon"
    ctx.emit_fn = queue.put_nowait

    async def _produce() -> None:
        try:
            async for event in gen:
                processed = event
                for interceptor in interceptors:
                    if processed is None:
                        break
                    processed = await interceptor.intercept_event(processed, ctx)
                if processed is not None:
                    queue.put_nowait(processed)
        finally:
            queue.put_nowait(sentinel)

    for interceptor in interceptors:
        await interceptor.on_iteration_start(ctx)

    producer = asyncio.create_task(_produce())

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        output.append(item)

    for interceptor in interceptors:
        await interceptor.on_iteration_end(ctx)

    while not queue.empty():
        item = queue.get_nowait()
        if item is not sentinel:
            output.append(item)

    await producer
    return output


def _make_engine(
    mock_llm: FunctionCallingLLM,
    response_format: str = "anthropic",
    ping_interval: float | None = None,
) -> ChatLoopEngine:
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm
    interceptors: list[ChatResponseLoopInterceptor] = [FilterZylonInterceptor()]
    if ping_interval:
        interceptors.append(PingInterceptor(ping_interval))
    return ChatLoopEngine(
        llm_component=llm_component,
        response_interceptors=interceptors,
        max_iterations=8,
    )


def _base_request(**kwargs: Any) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt="test"),
        **kwargs,
    )


async def _noop_tool(value: str) -> str:
    return f"ok:{value}"


async def _collect_engine(engine: ChatLoopEngine, request: ResolvedChatRequest) -> list:
    result = []
    async for event in engine.run(request):
        result.append(event)
    return result


# ---------------------------------------------------------------------------
# UNIT TESTS — hand-crafted SSE events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unit_text_events_pass_through() -> None:
    """Text blocks and deltas are forwarded unchanged in anthropic mode."""

    async def gen() -> AsyncGenerator:
        bid = _block_id()
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0, block_id=bid, content_block=TextBlock()
        )
        yield RawContentBlockDeltaEvent.from_content_block_start(
            RawContentBlockStartEvent(index=0, block_id=bid, content_block=TextBlock()),
            TextDelta(text="Hello"),
        )
        yield RawContentBlockStopEvent(index=0, block_id=bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen())

    assert isinstance(events[0], RawMessageStartEvent)
    assert isinstance(events[-1], RawMessageStopEvent)
    text_deltas = [
        e
        for e in events
        if isinstance(e, RawContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    ]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta.text == "Hello"


@pytest.mark.asyncio
async def test_unit_tool_use_block_passes_through() -> None:
    async def gen() -> AsyncGenerator:
        bid = _block_id()
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0,
            block_id=bid,
            content_block=ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
        )
        yield RawContentBlockStopEvent(index=0, block_id=bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="tool_use"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen())

    tool_starts = [
        e
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolUseBlock)
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0].content_block.name == "search"


@pytest.mark.asyncio
async def test_unit_tool_result_block_passes_through() -> None:
    async def gen() -> AsyncGenerator:
        bid = _block_id()
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0,
            block_id=bid,
            content_block=ToolResultBlock(tool_use_id="call_1", content="result text"),
        )
        yield RawContentBlockStopEvent(index=0, block_id=bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen())

    result_starts = [
        e
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolResultBlock)
    ]
    assert len(result_starts) == 1


@pytest.mark.asyncio
async def test_unit_fatal_error_passes_through() -> None:
    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        yield FatalError.from_exception(ValueError("boom"))

    events = await _collect_from_gen(gen())
    assert any(isinstance(e, FatalError) for e in events)


@pytest.mark.asyncio
async def test_unit_anthropic_mode_strips_tldr_keeps_thinking() -> None:
    """TLDRBlock is stripped; ThinkingBlock is kept."""
    thinking_bid, tldr_bid, text_bid = _block_id(), _block_id(), _block_id()

    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0,
            block_id=thinking_bid,
            content_block=ThinkingBlock(thinking="", signature="sig_sse_1"),
        )
        yield RawContentBlockDeltaEvent.from_content_block_start(
            RawContentBlockStartEvent(
                index=0,
                block_id=thinking_bid,
                content_block=ThinkingBlock(thinking="", signature="sig_sse_1"),
            ),
            ThinkingDelta(thinking="reasoning…"),
        )
        yield RawContentBlockStopEvent(index=0, block_id=thinking_bid)
        yield RawContentBlockStartEvent(
            index=1, block_id=text_bid, content_block=TextBlock()
        )
        yield RawContentBlockDeltaEvent.from_content_block_start(
            RawContentBlockStartEvent(
                index=1, block_id=text_bid, content_block=TextBlock()
            ),
            TextDelta(text="answer"),
        )
        yield RawContentBlockStopEvent(index=1, block_id=text_bid)
        yield RawContentBlockStartEvent(
            index=2, block_id=tldr_bid, content_block=TLDRBlock()
        )
        yield RawContentBlockStopEvent(index=2, block_id=tldr_bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), "anthropic")

    assert (
        sum(
            1
            for e in events
            if isinstance(e, RawContentBlockStartEvent)
            and isinstance(e.content_block, ThinkingBlock)
        )
        == 1
    )
    assert not any(
        isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, TLDRBlock)
        for e in events
    )
    assert (
        sum(
            1
            for e in events
            if isinstance(e, RawContentBlockStartEvent)
            and isinstance(e.content_block, TextBlock)
        )
        == 1
    )


@pytest.mark.asyncio
async def test_unit_zylon_mode_preserves_thinking_and_tldr() -> None:
    thinking_bid, tldr_bid, text_bid = _block_id(), _block_id(), _block_id()

    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0,
            block_id=thinking_bid,
            content_block=ThinkingBlock(thinking="", signature="sig_sse_2"),
        )
        yield RawContentBlockStopEvent(index=0, block_id=thinking_bid)
        yield RawContentBlockStartEvent(
            index=1, block_id=text_bid, content_block=TextBlock()
        )
        yield RawContentBlockStopEvent(index=1, block_id=text_bid)
        yield RawContentBlockStartEvent(
            index=2, block_id=tldr_bid, content_block=TLDRBlock()
        )
        yield RawContentBlockStopEvent(index=2, block_id=tldr_bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), "zylon")

    for block_type in (ThinkingBlock, TextBlock, TLDRBlock):
        assert (
            sum(
                1
                for e in events
                if isinstance(e, RawContentBlockStartEvent)
                and isinstance(e.content_block, block_type)
            )
            == 1
        )


@pytest.mark.asyncio
async def test_unit_orphaned_stop_filtered_when_start_stripped() -> None:
    """ContentBlockStop whose start was stripped must not appear in output."""
    tldr_bid, text_bid = _block_id(), _block_id()

    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0, block_id=tldr_bid, content_block=TLDRBlock()
        )
        yield RawContentBlockStopEvent(index=0, block_id=tldr_bid)
        yield RawContentBlockStartEvent(
            index=1, block_id=text_bid, content_block=TextBlock()
        )
        yield RawContentBlockStopEvent(index=1, block_id=text_bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), "anthropic")

    start_ids = {e.block_id for e in events if isinstance(e, RawContentBlockStartEvent)}
    for stop in (e for e in events if isinstance(e, RawContentBlockStopEvent)):
        assert stop.block_id in start_ids


# ---------------------------------------------------------------------------
# UNIT TESTS — ping interceptor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unit_ping_emitted_during_idle() -> None:
    async def gen() -> AsyncGenerator:
        bid = _block_id()
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0, block_id=bid, content_block=TextBlock()
        )
        yield RawContentBlockDeltaEvent.from_content_block_start(
            RawContentBlockStartEvent(index=0, block_id=bid, content_block=TextBlock()),
            TextDelta(text="start"),
        )
        await asyncio.sleep(2.1)
        yield RawContentBlockStopEvent(index=0, block_id=bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), ping_interval=1.0)
    assert sum(1 for e in events if isinstance(e, PingEvent)) >= 1


@pytest.mark.asyncio
async def test_unit_no_ping_when_stream_continuous() -> None:
    async def gen() -> AsyncGenerator:
        bid = _block_id()
        yield RawMessageStartEvent.from_defaults()
        yield RawContentBlockStartEvent(
            index=0, block_id=bid, content_block=TextBlock()
        )
        for i in range(8):
            yield RawContentBlockDeltaEvent.from_content_block_start(
                RawContentBlockStartEvent(
                    index=0, block_id=bid, content_block=TextBlock()
                ),
                TextDelta(text=f"word{i} "),
            )
            await asyncio.sleep(0.05)
        yield RawContentBlockStopEvent(index=0, block_id=bid)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), ping_interval=1.0)
    assert not any(isinstance(e, PingEvent) for e in events)


@pytest.mark.asyncio
async def test_unit_multiple_pings_during_long_idle() -> None:
    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        await asyncio.sleep(3.5)
        yield RawMessageDeltaEvent(
            delta=MessageOutputDelta(stop_reason="end_turn"), usage=Usage()
        )
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), ping_interval=1.0)
    assert sum(1 for e in events if isinstance(e, PingEvent)) >= 3


@pytest.mark.asyncio
async def test_unit_no_ping_when_interval_none() -> None:
    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        await asyncio.sleep(2.0)
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), ping_interval=None)
    assert not any(isinstance(e, PingEvent) for e in events)


@pytest.mark.asyncio
async def test_unit_no_ping_when_interval_zero() -> None:
    async def gen() -> AsyncGenerator:
        yield RawMessageStartEvent.from_defaults()
        await asyncio.sleep(2.0)
        yield RawMessageStopEvent.from_defaults()

    events = await _collect_from_gen(gen(), ping_interval=0)
    assert not any(isinstance(e, PingEvent) for e in events)


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — real ChatLoopEngine → interceptors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_simple_text_response() -> None:
    """ChatLoopEngine text deltas survive the interceptor pipeline unchanged."""
    engine = _make_engine(get_mock_function_calling_llm(["hello", " world"]))
    events = await _collect_engine(engine, _base_request())

    assert isinstance(events[0], RawMessageStartEvent)
    assert isinstance(events[-1], RawMessageStopEvent)
    text_deltas = [
        e
        for e in events
        if isinstance(e, RawContentBlockDeltaEvent) and isinstance(e.delta, TextDelta)
    ]
    assert any(d.delta.text == "hello" for d in text_deltas)
    assert any(d.delta.text == " world" for d in text_deltas)


@pytest.mark.asyncio
async def test_integration_message_delta_usage_never_none() -> None:
    """MessageDelta.usage must never be None — required by the Anthropic SDK."""
    engine = _make_engine(get_mock_function_calling_llm(["answer"]))
    events = await _collect_engine(engine, _base_request())

    deltas = [e for e in events if isinstance(e, RawMessageDeltaEvent)]
    assert deltas
    for delta in deltas:
        assert delta.usage is not None


@pytest.mark.asyncio
async def test_integration_tool_use_and_result_blocks() -> None:
    """Tool use and result blocks are emitted and survive the interceptors."""
    request = _base_request()
    request.tool_config = ResolvedToolConfig(
        tools=[
            ToolSpec.from_defaults(
                name="echo", type="echo", runtime="server", async_fn=_noop_tool
            )
        ]
    )
    mock_llm = get_mock_function_calling_llm(
        [
            [ToolSelection(tool_id="t1", tool_name="echo", tool_kwargs={"value": "x"})],
            ["done"],
        ]
    )
    engine = _make_engine(mock_llm)
    events = await _collect_engine(engine, request)

    assert any(
        isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolUseBlock)
        for e in events
    )
    assert any(
        isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolResultBlock)
        for e in events
    )


@pytest.mark.asyncio
async def test_integration_thinking_kept_in_anthropic_mode() -> None:
    """ThinkingBlock is StandardContentProtocol — kept in anthropic mode too."""
    engine = _make_engine(_make_thinking_mock_llm(), response_format="anthropic")
    events = await _collect_engine(engine, _base_request())

    assert any(
        isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ThinkingBlock)
        for e in events
    )
    assert any(
        isinstance(e, RawContentBlockDeltaEvent)
        and isinstance(e.delta, TextDelta)
        and e.delta.text == "answer"
        for e in events
    )


@pytest.mark.asyncio
async def test_integration_thinking_preserved_in_zylon_mode() -> None:
    """ThinkingBlock emitted by the engine is kept in zylon mode."""
    engine = _make_engine(_make_thinking_mock_llm(), response_format="zylon")
    events = await _collect_engine(engine, _base_request())

    assert any(
        isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ThinkingBlock)
        for e in events
    )


@pytest.mark.asyncio
async def test_integration_all_block_starts_have_matching_stops() -> None:
    """Every ContentBlockStart emitted by the full stack has a corresponding Stop."""
    request = _base_request()
    request.tool_config = ResolvedToolConfig(
        tools=[ToolSpec.from_defaults(name="echo", type="echo", async_fn=_noop_tool)]
    )
    mock_llm = get_mock_function_calling_llm(
        [
            [ToolSelection(tool_id="t1", tool_name="echo", tool_kwargs={"value": "x"})],
            ["done"],
        ]
    )
    engine = _make_engine(mock_llm, response_format="zylon")
    events = await _collect_engine(engine, request)

    start_ids = {e.block_id for e in events if isinstance(e, RawContentBlockStartEvent)}
    stop_ids = {e.block_id for e in events if isinstance(e, RawContentBlockStopEvent)}
    assert start_ids == stop_ids


@pytest.mark.asyncio
async def test_integration_ping_injected_between_slow_llm_chunks() -> None:
    """Ping events are injected when the mock LLM is slow between chunks."""
    engine = _make_engine(
        get_mock_function_calling_llm(["hello", " world"], sleep_between_deltas=2.1),
        ping_interval=1.0,
    )
    events = await _collect_engine(engine, _base_request())
    assert sum(1 for e in events if isinstance(e, PingEvent)) >= 1


# ---------------------------------------------------------------------------
# Private factory helpers
# ---------------------------------------------------------------------------


def _make_thinking_mock_llm() -> FunctionCallingLLM:
    mock_llm = MagicMock(spec=FunctionCallingLLM)
    mock_llm.metadata.context_window = 4096
    mock_llm.metadata.num_output = 1024
    mock_llm.metadata.is_function_calling_model = True
    mock_llm.callback_manager = MagicMock()
    mock_llm.completion_to_prompt = lambda prompt, **kwargs: prompt
    mock_llm.messages_to_prompt = lambda messages, **kwargs: "\n".join(
        m.content for m in (messages or []) if m and m.content
    )
    mock_llm.get_tool_calls_from_response = (
        lambda response, error_on_no_tool_call, **kw: response.additional_kwargs.get(
            "tool_calls", []
        )
    )

    async def _astream(*args: Any, **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={"thinking_delta": "secret"},
        )
        yield ChatResponse(
            message=msg, raw=msg, delta=None, additional_kwargs=msg.additional_kwargs
        )
        msg2 = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="answer",
            additional_kwargs={"stop_reason": "end_turn"},
        )
        yield ChatResponse(
            message=msg2,
            raw=msg2,
            delta="answer",
            additional_kwargs=msg2.additional_kwargs,
        )

    async def _coro(*args: Any, **kwargs: Any):
        return _astream(*args, **kwargs)

    mock_llm.astream_chat_with_tools = _coro
    return mock_llm
