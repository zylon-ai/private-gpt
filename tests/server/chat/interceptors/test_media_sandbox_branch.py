"""Image and audio attachments also land in the sandbox, as files.

Unlike documents this is not a branch: nothing is taken away. The blocks stay on
the message whatever happens, so the multimodal interceptor downstream still
sees exactly what it saw before — either sending them to a multimodal model or
replacing them with a description. All this interceptor adds is the bytes on
disk and a note saying where they are.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    MessageRole,
)

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
from private_gpt.server.chat.interceptors.media_file_interceptor import (
    MediaFilePreprocessingInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

PNG = b"\x89PNG\r\n\x1a\nfake"
MP3 = b"ID3fake"


def _placeholder(name: str) -> ToolSpec:
    return ToolSpec(name=name, type=f"{name}_v1")


def _resolved(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=f"{name}_v1",
        runtime="server",
        async_fn=AsyncMock(return_value=[]),
    )


_SANDBOX_TOOLS = [
    _resolved("bash_code_execution"),
    _resolved("describe_image"),
    _resolved("transcribe_audio"),
]


def _context(
    blocks: list[Any],
    tools: list[ToolSpec],
    role: MessageRole = MessageRole.USER,
) -> ChatInterceptorContext:
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=role, blocks=blocks)],
        system=ResolvedSystemConfig(model="model-a"),
        tool_config=ResolvedToolConfig(tools=tools),
        context=ResolvedContextConfig(container="session-42"),
    )
    state = ChatState(
        input=ChatInputState(request=request),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
    )
    return ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=lambda event: None,
    )


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.chat.preprocess.multimodal.max_concurrency = None
    return settings


def _file_service() -> MagicMock:
    file_service = MagicMock()
    file_service.exists = AsyncMock(return_value=False)
    file_service.put_file = AsyncMock()
    return file_service


def _interceptor(file_service: MagicMock) -> MediaFilePreprocessingInterceptor:
    return MediaFilePreprocessingInterceptor(
        file_service=file_service, settings=_settings()
    )


def _message(context: ChatInterceptorContext) -> ChatMessage:
    return context.state.input.request.messages[-1]


def _text(context: ChatInterceptorContext) -> str:
    return "\n".join(
        block.text for block in _message(context).blocks if getattr(block, "text", None)
    )


@pytest.mark.asyncio
async def test_an_image_is_written_to_uploads_and_the_block_is_kept() -> None:
    file_service = _file_service()
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")], _SANDBOX_TOOLS
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_awaited_once()
    call = file_service.put_file.await_args
    assert call.kwargs["scope_id"] == "session-42"
    assert call.kwargs["path"] == "image1.png"
    # The decisive assertion: ImageBlock re-encodes to base64 at construction,
    # so writing block.image verbatim would leave an unopenable file on disk.
    assert call.kwargs["content"] == PNG

    # Additive: the block survives, so a multimodal model still sees the image
    # and the downstream preprocessor still has something to work with.
    assert any(isinstance(block, ImageBlock) for block in _message(context).blocks)
    assert "/mnt/user-data/uploads/image1.png" in _text(context)


@pytest.mark.asyncio
async def test_audio_is_written_with_its_format_as_the_extension() -> None:
    file_service = _file_service()
    context = _context([AudioBlock(audio=MP3, format="mp3")], _SANDBOX_TOOLS)

    await _interceptor(file_service).intercept(context)

    call = file_service.put_file.await_args
    assert call.kwargs["path"] == "audio1.mp3"
    assert call.kwargs["content"] == MP3
    assert any(isinstance(block, AudioBlock) for block in _message(context).blocks)
    assert "/mnt/user-data/uploads/audio1.mp3" in _text(context)


@pytest.mark.asyncio
async def test_images_and_audio_upload_together() -> None:
    file_service = _file_service()
    context = _context(
        [
            ImageBlock(image=PNG, image_mimetype="image/png"),
            AudioBlock(audio=MP3, format="wav"),
            ImageBlock(image=PNG, image_mimetype="image/png"),
        ],
        _SANDBOX_TOOLS,
    )

    await _interceptor(file_service).intercept(context)

    paths = {call.kwargs["path"] for call in file_service.put_file.await_args_list}
    assert paths == {"image1.png", "image2.png", "audio1.wav"}


@pytest.mark.asyncio
async def test_the_note_warns_the_files_are_binary() -> None:
    """A model that can see the image will otherwise try to read the path."""
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")], _SANDBOX_TOOLS
    )

    await _interceptor(_file_service()).intercept(context)

    note = _text(context)
    assert "binary files" in note
    # Naming a tool in the note made the model reach for it every time; the
    # tool descriptions carry the applicability rules instead.
    assert "describe_image" not in note
    assert "transcribe_audio" not in note


@pytest.mark.asyncio
async def test_colliding_names_are_suffixed() -> None:
    file_service = _file_service()
    seen: set[str] = set()

    async def _exists(scope_id: str, path: str) -> bool:
        return path in seen

    async def _put_file(scope_id: str, path: str, content: bytes) -> None:
        seen.add(path)

    file_service.exists = AsyncMock(side_effect=_exists)
    file_service.put_file = AsyncMock(side_effect=_put_file)
    seen.add("image1.png")

    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")], _SANDBOX_TOOLS
    )
    await _interceptor(file_service).intercept(context)

    assert file_service.put_file.await_args.kwargs["path"] == "image1 (2).png"


@pytest.mark.asyncio
async def test_without_code_execution_nothing_is_uploaded() -> None:
    file_service = _file_service()
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")],
        [_resolved("semantic_search")],
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert _text(context) == ""


@pytest.mark.asyncio
async def test_an_unresolved_media_tool_does_not_count() -> None:
    """A placeholder means the pipeline has not built the tool yet."""
    file_service = _file_service()
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")],
        [_resolved("bash_code_execution"), _placeholder("describe_image")],
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_either_media_tool_is_enough() -> None:
    file_service = _file_service()
    context = _context(
        [AudioBlock(audio=MP3, format="mp3")],
        [_resolved("bash_code_execution"), _resolved("describe_image")],
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_message_without_media_is_left_alone() -> None:
    file_service = _file_service()
    context = _context([], _SANDBOX_TOOLS)
    before = _message(context).blocks

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
    assert _message(context).blocks == before


@pytest.mark.asyncio
async def test_a_failed_upload_leaves_the_request_untouched() -> None:
    """Nothing was taken away, so a failure costs the copy and nothing else."""
    file_service = _file_service()
    file_service.put_file = AsyncMock(side_effect=RuntimeError("disk full"))
    blocks = [ImageBlock(image=PNG, image_mimetype="image/png")]
    context = _context(blocks, _SANDBOX_TOOLS)

    await _interceptor(file_service).intercept(context)

    message = _message(context)
    assert any(isinstance(block, ImageBlock) for block in message.blocks)
    assert _text(context) == ""


@pytest.mark.asyncio
async def test_a_partial_failure_still_reports_what_landed() -> None:
    file_service = _file_service()
    calls = {"n": 0}

    async def _put_file(scope_id: str, path: str, content: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")

    file_service.put_file = AsyncMock(side_effect=_put_file)
    context = _context(
        [
            ImageBlock(image=PNG, image_mimetype="image/png"),
            ImageBlock(image=PNG, image_mimetype="image/png"),
        ],
        _SANDBOX_TOOLS,
    )

    await _interceptor(file_service).intercept(context)

    note = _text(context)
    assert "/mnt/user-data/uploads/image2.png" in note
    assert "upload failed: disk full" in note


@pytest.mark.asyncio
async def test_media_on_an_assistant_message_is_not_uploaded() -> None:
    """Only the last user message is uploaded, mirroring the document path."""
    file_service = _file_service()
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")],
        _SANDBOX_TOOLS,
        role=MessageRole.ASSISTANT,
    )

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_later_iterations_are_left_alone() -> None:
    """Re-running every iteration would fill the mount with copies."""
    file_service = _file_service()
    context = _context(
        [ImageBlock(image=PNG, image_mimetype="image/png")], _SANDBOX_TOOLS
    )
    context.state.runtime.iteration = 1

    await _interceptor(file_service).intercept(context)

    file_service.put_file.assert_not_awaited()
