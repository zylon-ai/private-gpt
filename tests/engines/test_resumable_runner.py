from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.engines.chat.checkpoint_store import ChatCheckpoint
from private_gpt.components.engines.chat.models.chat_state import ChatInputState
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.components.llm.custom.base import StructuredOutputsParams
from private_gpt.components.llm.models import ReasoningEffort


def test_original_input_restores_typed_llm_parameters() -> None:
    original_input = ChatInputState(
        request=ResolvedChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="hello")],
            system=ResolvedSystemConfig(prompt="test"),
        ),
        llm_kwargs={
            "reasoning_effort": ReasoningEffort.HIGH,
            "temperature": 0.2,
            "structured_outputs": StructuredOutputsParams(
                json_schema={"type": "object"},
            ),
        },
    )
    checkpoint = ChatCheckpoint(
        correlation_id="test",
        request_data={},
        original_input_data=ResumableChatRunner._dump_original_input(original_input),
        stream_type="chat_completion",
        metadata={},
        iteration=1,
    )

    restored = ResumableChatRunner._original_input(checkpoint)

    assert restored is not None
    assert restored.llm_kwargs.reasoning_effort is ReasoningEffort.HIGH
    assert restored.llm_kwargs.temperature == 0.2
    assert isinstance(restored.llm_kwargs.structured_outputs, StructuredOutputsParams)
    assert restored.llm_kwargs.structured_outputs.json_schema == {"type": "object"}
    assert restored.llm_kwargs.as_kwargs()["reasoning_effort"] is ReasoningEffort.HIGH
    assert isinstance(
        restored.llm_kwargs.as_kwargs()["structured_outputs"],
        StructuredOutputsParams,
    )
