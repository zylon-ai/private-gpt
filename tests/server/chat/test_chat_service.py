from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.events.models import (
    Event,
    FatalError,
    MessageOutputDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextDelta,
    Usage,
)
from private_gpt.server.chat.chat_service import ChatService, Completion, CompletionGen
from tests.fixtures.mock_injector import MockInjector


class _PromptInterceptorRequest(ChatRequestLoopInterceptor):
    def __init__(self, label: str) -> None:
        self._label = label

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        state = context.state.model_copy(deep=True)
        current = state.input.request.system.prompt or ""
        state.input.request.system.prompt = f"{current}{self._label}"
        context.set_state(state)


async def _basic_event_stream() -> AsyncGenerator[Event, None]:
    text_start = RawContentBlockStartEvent.from_text()
    yield RawMessageStartEvent.from_defaults()
    yield text_start
    yield RawContentBlockDeltaEvent.from_content_block_start(
        text_start,
        TextDelta(text="hello"),
    )
    yield RawContentBlockStopEvent.from_start(text_start)
    yield RawMessageDeltaEvent(
        delta=MessageOutputDelta(stop_reason="end_turn"),
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    yield RawMessageStopEvent.from_defaults()


async def _fatal_event_stream() -> AsyncGenerator[Event, None]:
    yield RawMessageStartEvent.from_defaults()
    yield FatalError.from_exception(RuntimeError("boom"))


def _mock_engine_for(stream: AsyncGenerator[Event, None]) -> MagicMock:
    mock_engine = MagicMock()
    mock_engine.run.return_value = stream
    return mock_engine


@pytest.mark.asyncio
async def test_chat_folds_loop_events(injector: MockInjector) -> None:
    service: ChatService = injector.get(ChatService)
    mock_engine = _mock_engine_for(_basic_event_stream())
    service._build_loop_engine = MagicMock(return_value=mock_engine)

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="hi", role=MessageRole.USER)],
        system=ResolvedSystemConfig(prompt="system"),
    )

    result = await service.chat(request)

    assert isinstance(result, Completion)
    assert result.response == "hello"
    assert result.stop_reason == "end_turn"
    assert result.usage is not None
    assert result.usage["input_tokens"] == 10
    assert result.usage["output_tokens"] == 20
    mock_engine.run.assert_called_once()


@pytest.mark.asyncio
async def test_stream_chat_returns_loop_generator(injector: MockInjector) -> None:
    service: ChatService = injector.get(ChatService)
    mock_engine = _mock_engine_for(_basic_event_stream())
    service._build_loop_engine = MagicMock(return_value=mock_engine)

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="hi", role=MessageRole.USER)],
        system=ResolvedSystemConfig(prompt="system"),
    )

    result = await service.stream_chat(request)

    assert isinstance(result, CompletionGen)
    events = [event async for event in result.events]
    assert events
    assert any(isinstance(event, RawMessageStopEvent) for event in events)


@pytest.mark.asyncio
async def test_validate_returns_error_when_no_user_text(
    injector: MockInjector,
) -> None:
    service: ChatService = injector.get(ChatService)

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="system", role=MessageRole.SYSTEM)],
        system=ResolvedSystemConfig(prompt="system"),
    )

    result = await service.validate(request)

    assert not result.valid
    assert result.errors is not None
    assert any(
        ("non-empty user text" in error) or ("No user message found" in error)
        for error in result.errors
    )


@pytest.mark.asyncio
async def test_chat_propagates_fatal_error_to_completion(
    injector: MockInjector,
) -> None:
    service: ChatService = injector.get(ChatService)
    mock_engine = _mock_engine_for(_fatal_event_stream())
    service._build_loop_engine = MagicMock(return_value=mock_engine)

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="hi", role=MessageRole.USER)],
        system=ResolvedSystemConfig(prompt="system"),
    )

    result = await service.chat(request)

    assert isinstance(result, Completion)
    assert result.exception is not None
    assert "boom" in str(result.exception)


@pytest.mark.asyncio
async def test_validate_runs_before_interceptors_in_order(
    injector: MockInjector,
) -> None:
    service: ChatService = injector.get(ChatService)

    mock_chain = MagicMock()
    mock_chain.request_interceptors = [
        _PromptInterceptorRequest("-one"),
        _PromptInterceptorRequest("-two"),
    ]
    mock_chain.response_interceptors = []
    service.chat_interceptor_service = MagicMock()
    service.chat_interceptor_service.get_chain.return_value = mock_chain

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="hello", role=MessageRole.USER)],
        system=ResolvedSystemConfig(prompt="start"),
    )

    validated = await service.validate(request)
    assert validated.valid


@pytest.mark.asyncio
async def test_validate_returns_valid_for_good_request(
    injector: MockInjector,
) -> None:
    service: ChatService = injector.get(ChatService)

    request = ResolvedChatRequest(
        messages=[ChatMessage(content="hello world", role=MessageRole.USER)],
        system=ResolvedSystemConfig(prompt="system"),
    )

    result = await service.validate(request)

    assert result.valid
    assert result.errors is None
