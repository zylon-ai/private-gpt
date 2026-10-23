"""Preprocessing interceptors must never leave a tool_use without a tool_result.

A failure escaping the document or multimodal preprocessing pipeline used to
propagate out of ``intercept`` after the ``tool_use`` block had been emitted,
which failed the whole chat with a dangling tool call in the stream.
"""

from collections.abc import AsyncIterator
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    DocumentProcessingResponse,
    DocumentProcessingStatus,
)
from private_gpt.components.chat.processors.chat_history.multimodality.models import (
    MultimodalProcessingResponse,
    MultimodalProcessingStatus,
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
    RawContentBlockStartEvent,
    ToolResultBlock,
    ToolUseBlock,
)
from private_gpt.server.chat.interceptors.document_file_interceptor import (
    DocumentFilePreprocessingInterceptor,
)
from private_gpt.server.chat.interceptors.multimodal_interceptor import (
    MultimodalRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

ReturnType = Literal["user_message", "tool_result"]


def _context() -> tuple[ChatInterceptorContext, list[Any]]:
    events: list[Any] = []
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(model="model-a"),
    )
    state = ChatState(
        input=ChatInputState(request=request),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
    )
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=events.append,
    )
    return context, events


def _settings(return_type: ReturnType) -> MagicMock:
    settings = MagicMock()
    for section in (
        settings.chat.preprocess.documents,
        settings.chat.preprocess.multimodal,
    ):
        section.max_concurrency = None
        section.return_type = return_type
        section.timeout_seconds = None
    return settings


def _tool_uses(events: list[Any]) -> list[ToolUseBlock]:
    return [
        e.content_block
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolUseBlock)
    ]


def _tool_results(events: list[Any]) -> list[ToolResultBlock]:
    return [
        e.content_block
        for e in events
        if isinstance(e, RawContentBlockStartEvent)
        and isinstance(e.content_block, ToolResultBlock)
    ]


def _assert_every_tool_use_has_error_result(events: list[Any]) -> list[str]:
    uses = _tool_uses(events)
    results = _tool_results(events)
    assert uses, "expected at least one tool_use to be emitted"
    assert len(results) == len(uses)
    for use, result in zip(uses, results, strict=True):
        assert result.tool_use_id == use.id
        assert result.is_error is True
        assert "boom" in str(result.content)
    return [use.id for use in uses]


def _assert_tool_messages_appended(
    context: ChatInterceptorContext, tool_ids: list[str], tool_name: str
) -> None:
    messages = context.state.input.request.messages
    assistant = [m for m in messages if m.role == MessageRole.ASSISTANT]
    tools = [m for m in messages if m.role == MessageRole.TOOL]
    assert len(assistant) == 1
    assert [
        call.tool_id for call in assistant[0].additional_kwargs["tool_calls"]
    ] == tool_ids
    assert [m.additional_kwargs["tool_call_id"] for m in tools] == tool_ids
    assert all(m.additional_kwargs["tool_call_name"] == tool_name for m in tools)


# ---------------------------------------------------------------------------
# Document preprocessing interceptor
# ---------------------------------------------------------------------------


async def _exploding_document_history(
    **_: Any,
) -> AsyncIterator[DocumentProcessingResponse]:
    yield DocumentProcessingResponse(
        processing_status=DocumentProcessingStatus(
            status="processing", doc_index=0, reference="deck.pptx"
        )
    )
    yield DocumentProcessingResponse(
        processing_status=DocumentProcessingStatus(
            status="processing", doc_index=1, reference="notes.docx"
        )
    )
    raise RuntimeError("boom: worker lost")


@pytest.mark.parametrize("return_type", ["user_message", "tool_result"])
@pytest.mark.asyncio
async def test_document_interceptor_closes_dangling_tool_uses_on_failure(
    monkeypatch: pytest.MonkeyPatch, return_type: ReturnType
) -> None:
    monkeypatch.setattr(
        "private_gpt.server.chat.interceptors.document_file_interceptor."
        "preprocess_document_history",
        _exploding_document_history,
    )
    interceptor = DocumentFilePreprocessingInterceptor(
        scheduler_factory=MagicMock(),
        file_service=MagicMock(),
        settings=_settings(return_type),
    )
    context, events = _context()

    await interceptor.intercept(context)  # must not raise

    tool_ids = _assert_every_tool_use_has_error_result(events)
    assert len(tool_ids) == 2
    if return_type == "tool_result":
        _assert_tool_messages_appended(context, tool_ids, "document_preprocessing")
    else:
        assert [m.role for m in context.state.input.request.messages] == [
            MessageRole.USER
        ]


# ---------------------------------------------------------------------------
# Multimodal preprocessing interceptor
# ---------------------------------------------------------------------------


async def _exploding_multimodal_history(
    **_: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    yield MultimodalProcessingResponse(
        processing_status=MultimodalProcessingStatus(status="processing", type="image")
    )
    raise RuntimeError("boom: request too large")


@pytest.mark.parametrize("return_type", ["user_message", "tool_result"])
@pytest.mark.asyncio
async def test_multimodal_interceptor_closes_dangling_tool_uses_on_failure(
    monkeypatch: pytest.MonkeyPatch, return_type: ReturnType
) -> None:
    monkeypatch.setattr(
        "private_gpt.server.chat.interceptors.multimodal_interceptor."
        "preprocess_multimodal_history",
        _exploding_multimodal_history,
    )
    llm_component = MagicMock()
    llm_component.get_config.return_value = MagicMock(support_image=0, support_audio=0)
    interceptor = MultimodalRequestInterceptor(
        llm_component=llm_component, settings=_settings(return_type)
    )
    context, events = _context()

    await interceptor.intercept(context)  # must not raise

    tool_ids = _assert_every_tool_use_has_error_result(events)
    assert len(tool_ids) == 1
    if return_type == "tool_result":
        _assert_tool_messages_appended(context, tool_ids, "multimodal_preprocessing")
    else:
        assert [m.role for m in context.state.input.request.messages] == [
            MessageRole.USER
        ]


@pytest.mark.asyncio
async def test_multimodal_interceptor_forwards_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-request multimodal timeout comes from settings, not a 100h default."""
    seen: dict[str, Any] = {}

    async def _capturing_history(
        **kwargs: Any,
    ) -> AsyncIterator[MultimodalProcessingResponse]:
        seen.update(kwargs)
        yield MultimodalProcessingResponse(chat_history=kwargs["chat_history"])

    monkeypatch.setattr(
        "private_gpt.server.chat.interceptors.multimodal_interceptor."
        "preprocess_multimodal_history",
        _capturing_history,
    )
    llm_component = MagicMock()
    llm_component.get_config.return_value = MagicMock(support_image=0, support_audio=0)
    settings = _settings("user_message")
    settings.chat.preprocess.multimodal.timeout_seconds = 12.5
    interceptor = MultimodalRequestInterceptor(
        llm_component=llm_component, settings=settings
    )
    context, _ = _context()

    await interceptor.intercept(context)

    assert seen["timeout"] == 12.5


async def _exploding_before_any_status(
    **_: Any,
) -> AsyncIterator[DocumentProcessingResponse]:
    raise RuntimeError("boom: before any attachment was announced")
    yield  # pragma: no cover


@pytest.mark.asyncio
async def test_document_interceptor_propagates_failure_with_nothing_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tool_use was emitted, so there is no tool_result to carry the error."""
    monkeypatch.setattr(
        "private_gpt.server.chat.interceptors.document_file_interceptor."
        "preprocess_document_history",
        _exploding_before_any_status,
    )
    interceptor = DocumentFilePreprocessingInterceptor(
        scheduler_factory=MagicMock(),
        file_service=MagicMock(),
        settings=_settings("tool_result"),
    )
    context, events = _context()

    with pytest.raises(RuntimeError, match="before any attachment"):
        await interceptor.intercept(context)

    assert events == []


async def _exploding_multimodal_before_any_status(
    **_: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    raise RuntimeError("boom: before any modality was announced")
    yield  # pragma: no cover


@pytest.mark.asyncio
async def test_multimodal_interceptor_propagates_failure_with_nothing_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "private_gpt.server.chat.interceptors.multimodal_interceptor."
        "preprocess_multimodal_history",
        _exploding_multimodal_before_any_status,
    )
    llm_component = MagicMock()
    llm_component.get_config.return_value = MagicMock(support_image=0, support_audio=0)
    interceptor = MultimodalRequestInterceptor(
        llm_component=llm_component, settings=_settings("tool_result")
    )
    context, events = _context()

    with pytest.raises(RuntimeError, match="before any modality"):
        await interceptor.intercept(context)

    assert events == []
