from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.components.engines.chat.utils.request_builder import (
    build_initial_context_stack,
)
from private_gpt.events.event_errors import Errors
from private_gpt.server.chat.interceptors.validator_request_interceptor import (
    ValidatorRequestInterceptor,
)
from private_gpt.settings.settings import LLMModelConfig
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


class DummyTokenizer:
    def __call__(
        self,
        texts: str | None = None,
        images: object | None = None,
        audios: object | None = None,
        **kwargs: object,
    ) -> list[int]:
        if texts is None:
            return []
        return texts.split()


async def _dummy_tool_async_fn(**kwargs: object) -> None:
    return None


def _build_model_config(
    *,
    support_reasoning: bool | None = True,
    support_image: int | None = None,
    support_audio: int | None = None,
) -> LLMModelConfig:
    return LLMModelConfig(
        name="model-x",
        mode="openai",
        support_reasoning=support_reasoning,
        support_image=support_image,
        support_audio=support_audio,
    )


def _build_interceptor(
    *,
    metadata: LLMMetadata | MultiModalLLMMetadata,
    config: LLMModelConfig,
    tokenizer: object | None = None,
) -> ValidatorRequestInterceptor:
    llm = MagicMock(spec=FunctionCallingLLM)
    llm.metadata = metadata

    llm_component = MagicMock()
    llm_component.get_llm.return_value = llm
    llm_component.get_config.return_value = config
    if tokenizer is not None:
        llm_component.get_tokenizer.return_value = tokenizer
    else:
        llm_component.get_tokenizer.side_effect = ValueError("no tokenizer")

    return ValidatorRequestInterceptor(llm_component=llm_component)


def _request_with_user_blocks(
    user_blocks: list[TextBlock | ImageBlock | AudioBlock],
    *,
    system_prompt: str | None = None,
    use_tools: bool = False,
    enable_reasoning: bool = False,
) -> ChatRequest:
    request = ResolvedChatRequest(
        system=ResolvedSystemConfig(
            prompt=[TextBlock(text=system_prompt)] if system_prompt else None
        ),
        messages=[
            ChatMessage(role=MessageRole.USER, blocks=user_blocks),
        ],
    )
    if use_tools:
        request.tool_config.tools = [
            ToolSpec.from_defaults(
                name="tool-a",
                type="tool-a",
                async_fn=_dummy_tool_async_fn,
            )
        ]
    request.thinking.enabled = enable_reasoning
    return request


async def _run_interceptor(
    interceptor: ValidatorRequestInterceptor,
    request: ChatRequest,
) -> None:
    state = ChatState(
        input=ChatInputState(
            request=request,
            context_stack=build_initial_context_stack(request),
            sampling_params=dict(request.sampling_params),
            llm_kwargs=dict(request.sampling_params),
        ),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
    )
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.VALIDATION,
        emit_fn=lambda _event: None,
    )
    await interceptor.intercept(context)


@pytest.mark.asyncio
async def test_rejects_tools_when_model_is_not_function_calling() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=False,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(),
    )
    request = _request_with_user_blocks([TextBlock(text="hello")], use_tools=True)

    with pytest.raises(Errors.InvalidRequest, match="does not support tool usage"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_reasoning_when_model_does_not_support_it() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(support_reasoning=False),
    )
    request = _request_with_user_blocks(
        [TextBlock(text="hello")], enable_reasoning=True
    )

    with pytest.raises(Errors.InvalidRequest, match="does not support reasoning"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_empty_user_message() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks([TextBlock(text="   ")])

    with pytest.raises(Errors.InvalidRequest, match="message cannot be empty"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_images_when_model_is_not_multimodal() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(support_image=1),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [
            TextBlock(text="image please"),
            ImageBlock(url="https://picsum.photos/200/300"),
        ]
    )

    with pytest.raises(Errors.InvalidRequest, match="does not support images"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_when_image_count_exceeds_model_limit() -> None:
    interceptor = _build_interceptor(
        metadata=MultiModalLLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(support_image=1),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [
            TextBlock(text="many images"),
            ImageBlock(url="https://picsum.photos/200/300"),
            ImageBlock(url="https://picsum.photos/201/301"),
        ]
    )

    with pytest.raises(Errors.InvalidRequest, match="a maximum of 1 images"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_audios_when_model_is_not_multimodal() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(support_audio=1),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [
            TextBlock(text="audio please"),
            AudioBlock(url="https://example.com/a.mp3"),
        ]
    )

    with pytest.raises(Errors.InvalidRequest, match="does not support audio"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_when_audio_count_exceeds_model_limit() -> None:
    interceptor = _build_interceptor(
        metadata=MultiModalLLMMetadata(
            is_function_calling_model=True,
            context_window=1024,
            num_output=128,
        ),
        config=_build_model_config(support_audio=1),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [
            TextBlock(text="many audios"),
            AudioBlock(url="https://example.com/a.mp3"),
            AudioBlock(url="https://example.com/b.mp3"),
        ]
    )

    with pytest.raises(Errors.InvalidRequest, match="a maximum of 1 audios"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_when_user_message_exceeds_token_limit() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=300,
            num_output=10,
        ),
        config=_build_model_config(),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks([TextBlock(text=" ".join(["w"] * 35))])

    with pytest.raises(Errors.RequestTooLarge, match="message length"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_when_system_prompt_exceeds_token_limit() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=300,
            num_output=10,
        ),
        config=_build_model_config(),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [TextBlock(text="small")],
        system_prompt=" ".join(["s"] * 35),
    )

    with pytest.raises(Errors.RequestTooLarge, match="system prompt length"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_rejects_when_user_plus_system_exceeds_token_limit() -> None:
    interceptor = _build_interceptor(
        metadata=LLMMetadata(
            is_function_calling_model=True,
            context_window=320,
            num_output=10,
        ),
        config=_build_model_config(),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [TextBlock(text=" ".join(["u"] * 30))],
        system_prompt=" ".join(["s"] * 30),
    )

    with pytest.raises(Errors.RequestTooLarge, match="and system prompt length"):
        await _run_interceptor(interceptor, request)


@pytest.mark.asyncio
async def test_accepts_valid_request() -> None:
    interceptor = _build_interceptor(
        metadata=MultiModalLLMMetadata(
            is_function_calling_model=True,
            context_window=4096,
            num_output=256,
        ),
        config=_build_model_config(
            support_reasoning=True,
            support_image=2,
            support_audio=2,
        ),
        tokenizer=DummyTokenizer(),
    )
    request = _request_with_user_blocks(
        [
            TextBlock(text="hello"),
            ImageBlock(url="https://picsum.photos/200/300"),
            AudioBlock(url="https://example.com/a.mp3"),
        ],
        system_prompt="system prompt",
        use_tools=True,
        enable_reasoning=True,
    )

    await _run_interceptor(interceptor, request)
