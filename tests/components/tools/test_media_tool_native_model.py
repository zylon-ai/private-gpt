"""Media tools are hidden when the request's model already handles the media."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata

from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.tools.processors.describe_image_processor import (
    DescribeImageProcessor,
)
from private_gpt.components.tools.processors.transcribe_audio_processor import (
    TranscribeAudioProcessor,
)


def _request(tool_name: str, model: str | None = "main") -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(
            model=model,
            prompt="",
            blob_visibility=BlobVisibilityMode.INTERNAL,
        ),
        tool_config=ResolvedToolConfig(
            tools=[ToolSpec(name=tool_name, type=f"{tool_name}_v1")],
        ),
        context=ResolvedContextConfig(correlation_id="c"),
    )


def _llm_component(*, image: int | None, audio: int | None) -> Mock:
    llm = SimpleNamespace(metadata=MultiModalLLMMetadata())
    config = SimpleNamespace(support_image=image, support_audio=audio)
    component = Mock()
    component.get_llm.return_value = llm
    component.get_config.return_value = config
    return component


def _builder(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        build_tool=AsyncMock(
            return_value=ToolSpec.from_defaults(
                name=name,
                type=f"{name}_v1",
                runtime="server",
                async_fn=AsyncMock(return_value=[]),
            )
        )
    )


def _settings(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        code_execution=SimpleNamespace(
            tools=SimpleNamespace(**{name: SimpleNamespace(enabled=True)})
        )
    )


@pytest.mark.asyncio
async def test_describe_image_is_hidden_when_the_model_sees_images() -> None:
    builder = _builder("describe_image")
    component = _llm_component(image=4, audio=None)
    request = _request("describe_image")

    processor = DescribeImageProcessor(builder, component, _settings("describe_image"))

    assert await processor.intercept(request)
    assert request.tool_config.tools == []
    builder.build_tool.assert_not_awaited()
    component.get_llm.assert_called_with("main")


@pytest.mark.asyncio
async def test_describe_image_is_kept_when_the_model_is_blind() -> None:
    builder = _builder("describe_image")
    request = _request("describe_image")

    processor = DescribeImageProcessor(
        builder, _llm_component(image=None, audio=1), _settings("describe_image")
    )

    assert await processor.intercept(request)
    assert [tool.name for tool in request.tool_config.tools] == ["describe_image"]
    builder.build_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_audio_is_hidden_when_the_model_hears_audio() -> None:
    builder = _builder("transcribe_audio")
    request = _request("transcribe_audio")

    processor = TranscribeAudioProcessor(
        builder, _llm_component(image=None, audio=1), _settings("transcribe_audio")
    )

    assert await processor.intercept(request)
    assert request.tool_config.tools == []
    builder.build_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcribe_audio_is_kept_when_the_model_is_deaf() -> None:
    builder = _builder("transcribe_audio")
    request = _request("transcribe_audio")

    processor = TranscribeAudioProcessor(
        builder, _llm_component(image=4, audio=None), _settings("transcribe_audio")
    )

    assert await processor.intercept(request)
    assert [tool.name for tool in request.tool_config.tools] == ["transcribe_audio"]
    builder.build_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_unknown_model_keeps_the_tool() -> None:
    builder = _builder("describe_image")
    component = _llm_component(image=4, audio=None)
    component.get_llm.side_effect = ValueError("Model 'x' not found")
    request = _request("describe_image")

    processor = DescribeImageProcessor(builder, component, _settings("describe_image"))

    assert await processor.intercept(request)
    builder.build_tool.assert_awaited_once()
