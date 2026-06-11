import copy
from collections.abc import Mapping
from typing import Any, Self

from llama_index.core.llms.llm import ToolSelection
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    TimelinePhase,
)
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.components.skills.models.skill_entities import (
    SkillVersionWithSkillEntity,
)


class ChatLoopInputState(BaseModel):
    """Request-scoped inputs for one loop run.

    ``request``       — the original immutable ChatRequest (restored each iteration).
    ``context_stack`` — the working stack for the current iteration, built by
                        ``build_initial_context_stack`` at loop start then enriched by
                        interceptors (skills, MCP tools, RAG, …).

    Interceptors append layers to ``context_stack`` via ``stack.append_layer()``.
    The engine materializes ``request`` from the stack just before calling the LLM.
    """

    request: ChatRequest
    context_stack: ContextStack = Field(default_factory=ContextStack)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if deep:
            return copy.deepcopy(self)
        return super().model_copy(update=update, deep=deep)


class ChatLoopRuntimeState(BaseModel):
    """Store runtime counters."""

    effective_token_limit: int | None = None
    tokenizer_fn: TokenizerFn | None = None

    iteration: int = 0
    max_iterations: int = 40
    cache: "ChatLoopRuntimeCache" = Field(
        default_factory=lambda: ChatLoopRuntimeCache()
    )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if deep:
            # Override the default deep copy behavior to use deepcopy
            # for the entire model, ensuring all nested structures
            # are copied correctly with types.
            # Otherwise, the types inside additional_kwargs dicts can get lost
            return copy.deepcopy(self)

        return super().model_copy(update=update, deep=deep)


class ChatLoopOutputState(BaseModel):
    """Store loop outputs and pending external handoffs."""

    stop_reason: str | None = None
    pending_external_tool_calls: list[ToolSelection] = Field(default_factory=list)


class ChatLoopTimelineEntry(BaseModel):
    """Capture one immutable timeline snapshot for debugging."""

    iteration: int
    phase: TimelinePhase
    conversation_size: int
    tool_count: int
    stop_reason: str | None = None


class SkillsRuntimeCache(BaseModel):
    """Validated/resolved skills cached during runtime."""

    entries: list[SkillVersionWithSkillEntity] = Field(default_factory=list)
    resources: dict[str, list[str]] = Field(
        default_factory=dict,
        description="skill_id → bundled file paths relative to the skill dir "
        "(SKILL.md excluded).",
    )


class ChatLoopRuntimeCache(BaseModel):
    """Runtime cache buckets for interceptors."""

    skill: SkillsRuntimeCache | None = None


class ChatLoopState(BaseModel):
    """Aggregate clonable loop state sections and history timeline."""

    input: ChatLoopInputState
    runtime: ChatLoopRuntimeState
    output: ChatLoopOutputState

    original_input: ChatLoopInputState | None = Field(default=None)
    timeline: list[ChatLoopTimelineEntry] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
