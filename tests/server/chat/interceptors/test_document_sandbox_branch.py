"""Attachments take one of two paths, decided by the request's tool list.

Without code execution they are converted to markdown and inlined into the user
message (the historical behaviour). With code execution *and* a resolved
``convert_documents`` they are written to the session uploads mount instead, and
the message gets a note naming their sandbox paths.
"""

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
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
from private_gpt.events.models import DocumentBlock
from private_gpt.server.chat.interceptors.document_file_interceptor import (
    DocumentFilePreprocessingInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


def _document(title: str, payload: bytes = b"%PDF-1.4") -> DocumentBlock:
    return DocumentBlock(
        title=title,
        source=DocumentBlock.Base64Source(
            data=base64.b64encode(payload).decode(),
            media_type="application/pdf",
        ),
    )


def _placeholder(name: str) -> ToolSpec:
    return ToolSpec(name=name, type=f"{name}_v1")


def _resolved(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=f"{name}_v1",
        runtime="server",
        async_fn=AsyncMock(return_value=[]),
    )


def _context(
    documents: list[DocumentBlock],
    tools: list[ToolSpec],
) -> tuple[ChatInterceptorContext, list[Any]]:
    events: list[Any] = []
    request = ResolvedChatRequest(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="summarise these",
                additional_kwargs={"document": documents},
            )
        ],
        system=ResolvedSystemConfig(model="model-a"),
        tool_config=ResolvedToolConfig(tools=tools),
        context=ResolvedContextConfig(container="session-42"),
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


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.chat.preprocess.documents.max_concurrency = None
    settings.chat.preprocess.documents.return_type = "user_message"
    return settings


def _file_service() -> MagicMock:
    file_service = MagicMock()
    file_service.exists = AsyncMock(return_value=False)
    file_service.put_file = AsyncMock()
    return file_service


def _scheduler_factory(text: str = "# Converted") -> MagicMock:
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = text
    return MagicMock(get=MagicMock(return_value=convert_service))


def _interceptor(file_service: MagicMock) -> DocumentFilePreprocessingInterceptor:
    return DocumentFilePreprocessingInterceptor(
        scheduler_factory=_scheduler_factory(),
        file_service=file_service,
        settings=_settings(),
    )


def _text(context: ChatInterceptorContext) -> str:
    message = context.state.input.request.messages[-1]
    return "\n".join(
        block.text for block in message.blocks if getattr(block, "text", None)
    )


_SANDBOX_TOOLS = [
    _resolved("bash_code_execution"),
    _resolved("convert_documents"),
]


@pytest.mark.asyncio
async def test_without_code_execution_documents_are_inlined() -> None:
    file_service = _file_service()
    context, _ = _context([_document("report.pdf")], tools=[_placeholder("web_search")])

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert "# Converted" in _text(context)
    assert "document" not in context.state.input.request.messages[-1].additional_kwargs


@pytest.mark.asyncio
async def test_with_code_execution_documents_are_uploaded() -> None:
    file_service = _file_service()
    context, _ = _context(
        [_document("report.pdf"), _document("data.csv")], tools=_SANDBOX_TOOLS
    )

    await _interceptor(file_service).intercept(context)

    assert file_service.put_file.await_count == 2
    assert {
        call.kwargs["scope_id"] for call in file_service.put_file.await_args_list
    } == {"session-42"}
    assert sorted(
        call.kwargs["path"] for call in file_service.put_file.await_args_list
    ) == ["data.csv", "report.pdf"]

    note = _text(context)
    assert "/mnt/user-data/uploads/report.pdf" in note
    assert "/mnt/user-data/uploads/data.csv" in note
    # The note says where the files are and nothing else. Naming a tool here
    # made the model reach for it on every attachment, whatever was asked.
    assert "convert_documents" not in note
    # The bytes are on disk now; the markdown must not also be inlined.
    assert "# Converted" not in note
    assert "document" not in context.state.input.request.messages[-1].additional_kwargs


@pytest.mark.asyncio
async def test_colliding_titles_get_a_windows_style_suffix() -> None:
    file_service = _file_service()
    taken: set[str] = set()

    async def exists(scope_id: str, candidate: str) -> bool:
        return candidate in taken

    async def put_file(scope_id: str, path: str, content: bytes, **_: Any) -> None:
        taken.add(path)

    file_service.exists = AsyncMock(side_effect=exists)
    file_service.put_file = AsyncMock(side_effect=put_file)

    context, _ = _context(
        [_document("report.pdf", b"a"), _document("report.pdf", b"b")],
        tools=_SANDBOX_TOOLS,
    )

    await _interceptor(file_service).intercept(context)

    assert taken == {"report.pdf", "report (2).pdf"}
    assert "/mnt/user-data/uploads/report (2).pdf" in _text(context)


@pytest.mark.asyncio
async def test_code_execution_without_a_converter_falls_back_to_inlining() -> None:
    """Uploading with no way to read the content back would degrade silently."""
    file_service = _file_service()
    context, _ = _context(
        [_document("report.pdf")], tools=[_resolved("bash_code_execution")]
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert "# Converted" in _text(context)


@pytest.mark.asyncio
async def test_an_unresolved_converter_does_not_count() -> None:
    """A placeholder means the processor dropped it — e.g. disabled by settings."""
    file_service = _file_service()
    context, _ = _context(
        [_document("report.pdf")],
        tools=[_resolved("bash_code_execution"), _placeholder("convert_documents")],
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert "# Converted" in _text(context)


@pytest.mark.asyncio
async def test_a_total_upload_failure_falls_back_to_inlining() -> None:
    file_service = _file_service()
    file_service.put_file = AsyncMock(side_effect=RuntimeError("storage down"))
    context, _ = _context([_document("report.pdf")], tools=_SANDBOX_TOOLS)

    await _interceptor(file_service).intercept(context)

    assert "# Converted" in _text(context)


@pytest.mark.asyncio
async def test_a_partial_upload_failure_is_reported_in_the_note() -> None:
    file_service = _file_service()
    calls = {"n": 0}

    async def put_file(scope_id: str, path: str, content: bytes, **_: Any) -> None:
        calls["n"] += 1
        if path == "bad.pdf":
            raise RuntimeError("storage down")

    file_service.put_file = AsyncMock(side_effect=put_file)
    context, _ = _context(
        [_document("good.pdf"), _document("bad.pdf")], tools=_SANDBOX_TOOLS
    )

    await _interceptor(file_service).intercept(context)

    note = _text(context)
    assert "/mnt/user-data/uploads/good.pdf" in note
    assert "bad.pdf — upload failed: storage down" in note


@pytest.mark.asyncio
async def test_later_iterations_are_left_alone() -> None:
    file_service = _file_service()
    context, _ = _context([_document("report.pdf")], tools=_SANDBOX_TOOLS)
    context.state.runtime.iteration = 1

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert "document" in context.state.input.request.messages[-1].additional_kwargs
