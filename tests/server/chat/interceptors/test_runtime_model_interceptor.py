from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, LLMMetadata, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
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
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.server.chat.interceptors.runtime_model_interceptor import (
    RuntimeModelRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


def _context(
    *,
    phase: InterceptorPhase,
    model_id: str | None = "model-a",
) -> tuple[ChatInterceptorContext, MagicMock, MagicMock]:
    tokenizer = MagicMock()
    context_llm = get_mock_function_calling_llm(["ok"])
    llm = MagicMock()
    llm.metadata = LLMMetadata(context_window=131_072, num_output=4_096)
    llm_component = MagicMock()
    llm_component.get_llm.return_value = llm
    llm_component.get_tokenizer.return_value = tokenizer
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(model=model_id),
    )
    state = ChatState(
        input=ChatInputState(request=request),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
    )
    return (
        ChatInterceptorContext(
            state=state,
            llm=context_llm,
            phase=phase,
            emit_fn=lambda _event: None,
        ),
        llm_component,
        tokenizer,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [InterceptorPhase.VALIDATION, InterceptorPhase.BEFORE_ITERATION],
)
async def test_runtime_model_interceptor_hydrates_model_runtime(
    phase: InterceptorPhase,
) -> None:
    context, llm_component, tokenizer = _context(phase=phase)

    await RuntimeModelRequestInterceptor(llm_component).intercept(context)

    assert context.state.runtime.model_id == "model-a"
    assert context.state.runtime.effective_token_limit == 126_720
    assert context.state.runtime.tokenizer_fn is tokenizer
    llm_component.get_tokenizer.assert_called_once_with("model-a")


@pytest.mark.asyncio
async def test_runtime_model_interceptor_rebuilds_process_local_fields_from_model_id() -> (
    None
):
    context, llm_component, tokenizer = _context(
        phase=InterceptorPhase.BEFORE_ITERATION
    )
    context.state.runtime.model_id = "persisted-model"

    await RuntimeModelRequestInterceptor(llm_component).intercept(context)

    assert context.state.runtime.effective_token_limit == 126_720
    assert context.state.runtime.tokenizer_fn is tokenizer
    llm_component.get_tokenizer.assert_called_once_with("persisted-model")


@pytest.mark.asyncio
async def test_runtime_model_interceptor_skips_already_hydrated_runtime() -> None:
    context, llm_component, tokenizer = _context(
        phase=InterceptorPhase.BEFORE_ITERATION
    )
    context.state.runtime.model_id = "model-a"
    context.state.runtime.effective_token_limit = 100_000
    context.state.runtime.tokenizer_fn = tokenizer

    await RuntimeModelRequestInterceptor(llm_component).intercept(context)

    assert context.state.runtime.effective_token_limit == 100_000
    llm_component.get_tokenizer.assert_not_called()


def test_checkpoint_payload_persists_only_model_identity() -> None:
    context, _, tokenizer = _context(phase=InterceptorPhase.BEFORE_ITERATION)
    context.state.runtime.model_id = "persisted-model"
    context.state.runtime.effective_token_limit = 100_000
    context.state.runtime.tokenizer_fn = tokenizer

    payload = ResumableChatRunner._checkpoint_payload(context.state)
    serialized = payload.model_dump(mode="json")

    assert serialized["model_id"] == "persisted-model"
    assert "tokenizer_fn" not in serialized
    assert "effective_token_limit" not in serialized
