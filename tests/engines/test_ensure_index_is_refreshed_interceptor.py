"""Tests for EnsureIndexIsRefreshedInterceptor block index rewriting."""

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
)
from private_gpt.components.engines.chat.interceptors.ensure_index_is_refreshed_interceptor import (
    EnsureIndexIsRefreshedInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.events.models import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TextDelta,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


def _make_context(next_block_count: int) -> ChatInterceptorContext:
    state = ChatState(
        input=ChatInputState(
            request=ResolvedChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="hi")]
            )
        ),
        runtime=ChatRuntimeState(next_block_count=next_block_count),
        output=ChatOutputState(),
    )
    return ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.STREAMING,
        emit_fn=lambda _: None,
    )


@pytest.mark.asyncio
async def test_rewrites_indexes_relative_to_the_iteration_block_count() -> None:
    interceptor = EnsureIndexIsRefreshedInterceptor()
    context = _make_context(next_block_count=3)
    await interceptor.on_iteration_start(context)

    start = RawContentBlockStartEvent(
        index=0, block_id="b1", content_block=TextBlock(text="")
    )
    delta = RawContentBlockDeltaEvent.from_content_block_start(
        start, TextDelta(text="hello")
    )
    stop = RawContentBlockStopEvent.from_start(start)

    out_start = await interceptor.intercept_event(start, context)
    out_delta = await interceptor.intercept_event(delta, context)
    out_stop = await interceptor.intercept_event(stop, context)

    assert [e.index if e else None for e in (out_start, out_delta, out_stop)] == [
        3,
        3,
        3,
    ]
    assert context.state.runtime.next_block_count == 4
    # Inputs are never mutated in place.
    assert [start.index, delta.index, stop.index] == [0, 0, 0]


@pytest.mark.asyncio
async def test_returns_events_unchanged_when_index_is_already_correct() -> None:
    interceptor = EnsureIndexIsRefreshedInterceptor()
    context = _make_context(next_block_count=0)
    await interceptor.on_iteration_start(context)

    start = RawContentBlockStartEvent(
        index=0, block_id="b1", content_block=TextBlock(text="")
    )
    delta = RawContentBlockDeltaEvent.from_content_block_start(
        start, TextDelta(text="hello")
    )

    assert await interceptor.intercept_event(start, context) is start
    assert await interceptor.intercept_event(delta, context) is delta
