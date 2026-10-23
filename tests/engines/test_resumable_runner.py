from datetime import UTC, datetime

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.engines.chat.checkpoint_store import ChatCheckpoint
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeCache,
    ChatRuntimeState,
    ChatState,
    SkillsRuntimeCache,
)
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.components.llm.custom.base import StructuredOutputsParams
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.skills.models.skill_entities import (
    SkillEntity,
    SkillFrontmatter,
    SkillVersionEntity,
    SkillVersionWithSkillEntity,
)


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


def test_runtime_state_deepcopy_shares_unpickleable_tokenizer() -> None:
    import threading

    lock = threading.Lock()

    def tokenizer(text: str) -> list[int]:
        with lock:
            return [ord(char) for char in text]

    runtime = ChatRuntimeState(
        model_id="mock",
        tokenizer_fn=tokenizer,
        cache=ChatRuntimeCache(),
    )
    copied = runtime.model_copy(deep=True)
    assert copied.tokenizer_fn is tokenizer
    assert copied.model_id == "mock"
    assert tokenizer("ab") == [97, 98]


def test_runtime_cache_roundtrips_through_checkpoint() -> None:
    now = datetime.now(UTC)
    entry = SkillVersionWithSkillEntity(
        skill=SkillEntity(
            id="skill-creator",
            collection="col",
            display_title="skill-creator",
            source="zylon",
            loading="lazy",
            readonly=True,
            created_at=now,
            updated_at=now,
        ),
        version=SkillVersionEntity(
            id="ver-creator",
            skill_id="skill-creator",
            version="1",
            frontmatter=SkillFrontmatter(
                name="skill-creator", description="Create skills"
            ),
            storage_prefix="skills/creator",
            created_at=now,
        ),
    )
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt="test"),
    )
    state = ChatState(
        input=ChatInputState(request=request),
        runtime=ChatRuntimeState(
            model_id="mock",
            effective_token_limit=2048,
            cache=ChatRuntimeCache(
                skill=SkillsRuntimeCache(
                    entries=[entry],
                    resources={"skill-creator": ["scripts/init.py"]},
                )
            ),
        ),
        output=ChatOutputState(),
    )

    checkpoint = ChatCheckpoint(
        correlation_id="test",
        request_data=request.model_dump(mode="json"),
        runtime_data=ResumableChatRunner._dump_runtime(state),
        runtime_cache_data=ResumableChatRunner._dump_runtime_cache(state),
        stream_type="chat_completion",
        metadata={},
        iteration=1,
    )
    restored_cache = ResumableChatRunner._runtime_cache(checkpoint)
    restored_runtime = ResumableChatRunner._runtime(checkpoint)

    assert restored_cache is not None
    assert restored_cache.skill is not None
    assert restored_cache.skill.entries[0].version.frontmatter.name == "skill-creator"
    assert restored_cache.skill.resources == {"skill-creator": ["scripts/init.py"]}
    assert restored_runtime is not None
    assert restored_runtime.cache is not None
    assert restored_runtime.cache.skill is not None
    assert restored_runtime.model_id == "mock"
    assert restored_runtime.effective_token_limit == 2048
    assert restored_runtime.tokenizer_fn is None
