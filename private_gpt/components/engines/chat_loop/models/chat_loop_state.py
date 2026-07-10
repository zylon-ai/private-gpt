from collections.abc import Mapping
from typing import Any, Self

from llama_index.core.llms.llm import ToolSelection
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    TimelinePhase,
)
from private_gpt.components.llm.llm_helper import AsyncTokenizerFn, TokenizerFn
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


class ChatLoopRuntimeState(BaseModel):
    """Store runtime counters."""

    effective_token_limit: int | None = None
    tokenizer_fn: TokenizerFn | AsyncTokenizerFn | None = None

    iteration: int = 0
    max_iterations: int = 40
    cache: "ChatLoopRuntimeCache" = Field(
        default_factory=lambda: ChatLoopRuntimeCache()
    )


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

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:

        # Temporarily detach fields that must not be deep-copied:
        # - tokenizer_fn: HF tokenizer is not safely copyable and expensive
        # - context_stack: pydantic frozen — all mutations return new instances
        # - original_input: set once at loop init, never mutated
        tokenizer_fn = self.runtime.tokenizer_fn
        context_stack = self.input.context_stack
        original_input = self.original_input

        self.runtime.tokenizer_fn = None
        self.input.context_stack = ContextStack()
        self.original_input = None

        try:
            copied = super().model_copy(update=update, deep=deep)
        finally:
            self.runtime.tokenizer_fn = tokenizer_fn
            self.input.context_stack = context_stack
            self.original_input = original_input

        copied.runtime.tokenizer_fn = tokenizer_fn
        copied.input.context_stack = context_stack
        copied.original_input = original_input

        return copied
