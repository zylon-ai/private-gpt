"""Job-native async chat loop engine — each method is one job phase.

Parallel implementation of ChatLoopEngine: shares the same Pydantic state
models but exposes a job-friendly API instead of a while loop. Intended to
replace ChatLoopEngine once regression tests confirm identical event output.

Job graph::

    run_start
        └── run_iteration(0)
                ├── COMPLETED → close_chat_job(stop_reason)
                ├── CONTINUE  → run_iteration(N+1)
                └── WAITING   → IterationScheduler.on_waiting
                                    └── (all tools done) → resume("after_tool")
                                                               └── run_iteration(N+1)
"""

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import AsyncBaseTool, ToolSelection, adapt_to_async_tool
from pydantic import BaseModel, Field

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
    ToolSpec,
)
from private_gpt.components.container_registry import ContainerRegistry
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.chat_engine_interface import (
    ChatEngineExecution,
)
from private_gpt.components.engines.chat.chat_runner import ChatRunner
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.interceptors.ensure_index_is_refreshed_interceptor import (
    EnsureIndexIsRefreshedInterceptor,
)
from private_gpt.components.engines.chat.interceptors.ensure_timestamp_in_content_blocks_interceptors import (
    EnsureTimestampInContentBlocksInterceptor,
)
from private_gpt.components.engines.chat.interceptors.ensure_tools_are_flatten_interceptor import (
    EnsureToolAreFlattenInterceptor,
)
from private_gpt.components.engines.chat.interceptors.restore_stateless_input_interceptor import (
    RestoreStatelessInputInterceptorRequest,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
    TimelinePhase,
)
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
    ChatStatus,
    ChatTimelineEntry,
)
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
)
from private_gpt.components.engines.chat.utils.request_builder import (
    build_initial_context_stack,
    build_request_from_context_stack,
)
from private_gpt.components.engines.chat.utils.tool_utils import (
    select_tool_names,
)
from private_gpt.components.llm.custom.base import StructuredOutputsParams, ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.llm.priorities import DefinedPriorities
from private_gpt.components.tools.processors.base import _session_id
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionRequest,
    ToolExecutionResponse,
    build_tool_execution_context,
)
from private_gpt.components.tools.tool_scheduler import (
    BaseToolScheduler,
    LocalToolScheduler,
)
from private_gpt.events.models import (
    Container,
    Event,
    InputJSONDelta,
    MessageOutputDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    StopReasonEnum,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from private_gpt.components.streaming.tasks.chat_scheduler import (
        BaseChatScheduler,
    )
from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    _coerce_kwargs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data classes (mirrors of ChatLoopEngine — no shared import so that
# AsyncChatEngine stays independent once ChatLoopEngine is deleted)
# ---------------------------------------------------------------------------


class _ToolExecutionStatus:
    EXECUTED = "executed"
    NOT_EXECUTED = "not_executed"
    PENDING = "pending"
    NOT_FOUND = "not_found"


class _IterationCheckpoint:
    START = "start"
    BEFORE_ITERATION = "before_iteration"
    TOOLS = "tools"
    AFTER_ITERATION = "after_iteration"
    CLOSE = "close"


class IterationCheckpointPayload(BaseModel):
    pending_async_tools: dict[str, str] = Field(default_factory=dict)
    tool_responses: list[ToolExecutionResponse] = Field(default_factory=list)
    pending_external_tool_calls: list[ToolSelection] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    has_input_usage: bool = False
    has_output_usage: bool = False


class AsyncChatCheckpoint(BaseModel):
    checkpoint: str
    input: ChatInputState
    iteration: int = 0
    next_block_count: int = 0
    payload: IterationCheckpointPayload = Field(
        default_factory=IterationCheckpointPayload
    )


@dataclass
class _CheckpointContext:
    checkpoint: str
    stop_reason: str | None = None
    context_stack: ContextStack | None = None
    payload: IterationCheckpointPayload = field(
        default_factory=IterationCheckpointPayload
    )


@dataclass
class _ToolExecutionResult:
    status: str
    tool_selection: ToolSelection | None = None
    async_handle: str | None = None


@dataclass
class _Run:
    """Hold mutable runtime references for one phase invocation."""

    state: ChatState
    llm: FunctionCallingLLM
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    has_input_usage: bool = False
    has_output_usage: bool = False
    block_count: int = 0
    hooks: ExecutionHooks = field(default_factory=ExecutionHooks)


@dataclass
class _ToolDeltaState:
    active_tool_block: RawContentBlockStartEvent | None = None
    active_tool_raw_id: str | None = None
    tool_id_map: dict[str, str] = field(default_factory=dict)
    finished_tool_raw_ids: set[str] = field(default_factory=set)
    last_serialized: dict[str, str] = field(default_factory=dict)
    pending_tasks: list[asyncio.Task[_ToolExecutionResult]] = field(
        default_factory=list
    )
    tool_semaphore: asyncio.Semaphore | None = None


@dataclass
class _StreamDeltaState:
    active_block: RawContentBlockStartEvent | None = None
    active_block_kind: Literal["text", "thinking"] | None = None
    tool_state: _ToolDeltaState = field(default_factory=_ToolDeltaState)


class _Done:
    """Sentinel: producer finished."""


class EventChannel(ABC):
    """Sink for events produced by the engine."""

    @abstractmethod
    def emit(self, event: Event) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class LocalEventChannel(EventChannel):
    """asyncio.Queue-backed channel for local (in-process) execution."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | _Done] = asyncio.Queue()

    def emit(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def close(self) -> None:
        self._queue.put_nowait(_Done())

    async def stream(
        self, task: asyncio.Task[Any] | None = None
    ) -> AsyncGenerator[Event, None]:
        try:
            while True:
                item = await self._queue.get()
                if isinstance(item, _Done):
                    break
                yield item
        finally:
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                with suppress(asyncio.CancelledError):
                    await task


@dataclass
class _EventHandler(EventChannel):
    queue: asyncio.Queue[Event | _Done]

    def emit(self, event: Event) -> None:
        self.queue.put_nowait(event)

    async def close(self) -> None:
        self.queue.put_nowait(_Done())

    async def stream(self, producer: asyncio.Task[Any]) -> AsyncGenerator[Event, None]:
        try:
            while True:
                item = await self.queue.get()
                if isinstance(item, _Done):
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer


# ---------------------------------------------------------------------------
# AsyncChatEngine
# ---------------------------------------------------------------------------


class AsyncChatEngine:
    """Job-native chat loop engine.

    Each public method corresponds to exactly one ARQ job phase. The engine
    never drives the next phase itself — job code reads ``final_state.output.status``
    and dispatches the next job accordingly.

    status values returned:
      CONTINUE  — phase complete; job should dispatch the next phase
      COMPLETED — no more LLM iterations needed; job should dispatch close_chat_job
      WAITING   — async tools are pending; job should call IterationScheduler.on_waiting
    """

    def __init__(
        self,
        llm_component: LLMComponent,
        chat_scheduler: "BaseChatScheduler",
        request_interceptors: list[ChatRequestLoopInterceptor] | None = None,
        response_interceptors: list[ChatResponseLoopInterceptor] | None = None,
        max_iterations: int = 40,
        container_registry: ContainerRegistry | None = None,
        tool_scheduler: BaseToolScheduler | None = None,
        tool_interceptors: list[ToolExecutionInterceptor] | None = None,
        runner: ChatRunner | None = None,
    ) -> None:
        self._llm_component = llm_component
        self._request_interceptors = [
            RestoreStatelessInputInterceptorRequest(),
            *(request_interceptors or []),
            EnsureToolAreFlattenInterceptor(),
        ]
        self._response_interceptors = [
            *(response_interceptors or []),
            EnsureIndexIsRefreshedInterceptor(),
            EnsureTimestampInContentBlocksInterceptor(),
        ]
        self._max_iterations = max_iterations
        self._container_registry = container_registry
        self._tool_scheduler: BaseToolScheduler = tool_scheduler or LocalToolScheduler()
        self._tool_interceptors = tool_interceptors or []
        self._chat_scheduler = chat_scheduler
        self._runner = runner
        self._checkpoint_handlers = {
            _IterationCheckpoint.START: self._execute_start_checkpoint,
            _IterationCheckpoint.BEFORE_ITERATION: self._execute_before_iteration_checkpoint,
            _IterationCheckpoint.TOOLS: self._process_tools_checkpoint,
            _IterationCheckpoint.AFTER_ITERATION: self._execute_after_iteration_checkpoint,
            _IterationCheckpoint.CLOSE: self._execute_close_checkpoint,
        }

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
        *,
        channel: EventChannel,
    ) -> ChatState:
        """Run from start through the loop until WAITING or COMPLETED."""
        state = await self._execute_start(request, hooks, channel)
        return await self._continue_loop(
            state.input.request,
            state.runtime.iteration,
            state.runtime.next_block_count,
            IterationCheckpointPayload(),
            hooks,
            channel,
        )

    async def resume(
        self,
        checkpoint: AsyncChatCheckpoint,
        hooks: ExecutionHooks | None = None,
        *,
        channel: EventChannel,
    ) -> ChatState:
        """Route to the saved checkpoint, then continue the loop."""
        context = self._build_checkpoint_context(
            checkpoint=checkpoint.checkpoint,
            payload=checkpoint.payload,
            context_stack=checkpoint.input.context_stack,
        )
        handler = self._resolve_checkpoint_handler(checkpoint.checkpoint)
        state = await handler(
            checkpoint.input.request,
            checkpoint.iteration,
            checkpoint.next_block_count,
            channel,
            hooks,
            context,
        )
        new_payload = IterationCheckpointPayload(
            total_input_tokens=state.runtime.total_input_tokens,
            total_output_tokens=state.runtime.total_output_tokens,
            has_input_usage=state.runtime.has_input_usage,
            has_output_usage=state.runtime.has_output_usage,
        )
        if state.output.status == ChatStatus.WAITING:
            return state
        elif state.output.status == ChatStatus.COMPLETED:
            close_state = await self._execute_close(
                state.input.request,
                state.output.stop_reason,
                new_payload,
                hooks,
                channel,
            )
            if state.output.pending_external_tool_calls:
                close_state = close_state.model_copy(deep=True)
                close_state.output.pending_external_tool_calls = list(
                    state.output.pending_external_tool_calls
                )
            return close_state
        return await self._continue_loop(
            state.input.request,
            state.runtime.iteration,
            state.runtime.next_block_count,
            new_payload,
            hooks,
            channel,
        )

    async def run(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
        *,
        runner: ChatRunner | None = None,
    ) -> ChatEngineExecution:
        """Submit execution through the transport-independent resumable runner."""
        runner = runner or self._runner
        if runner is None:
            raise RuntimeError("AsyncChatEngine requires a ResumableChatRunner")
        execution_id, broker_events = await runner.submit(
            request_data=request.model_dump(mode="json"),
            stream_type="chat_completion",
            metadata={
                "message_count": len(request.messages),
                "thinking_enabled": request.thinking.enabled,
            },
            execution_id=getattr(request.context, "correlation_id", None),
        )

        async def _events() -> AsyncGenerator[Event, None]:
            completed = False
            try:
                async for event in broker_events:
                    yield event
                completed = True
            finally:
                if not completed:
                    await runner.cancel(execution_id)

        return ChatEngineExecution(
            events=_events(), final_state_task=None, execution_id=execution_id
        )

    async def execute_scheduled_start(
        self,
        *,
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        runner = self._require_runner()
        await runner.start(
            engine=self,
            execution_id=execution_id,
            request_data=request_data,
            stream_type=stream_type,
            metadata=metadata,
        )

    async def execute_scheduled_resume(
        self, *, execution_id: str, checkpoint_id: str
    ) -> None:
        runner = self._require_runner()
        await runner.resume(
            engine=self,
            execution_id=execution_id,
            checkpoint_id=checkpoint_id,
        )

    async def record_callback(
        self, *, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> None:
        runner = self._require_runner()
        await runner.callback(
            execution_id=execution_id,
            tool_id=tool_id,
            result=result,
        )

    async def validate(self, request: ResolvedChatRequest) -> None:
        await self.run_interceptor_phase(
            run=self.initialize_run(request),
            phase=InterceptorPhase.VALIDATION,
            interceptors=self._request_interceptors,
        )

    async def cancel(self, correlation_id: str) -> bool:
        if self._runner is not None:
            return bool(await self._runner.cancel(correlation_id))
        return bool(await self._chat_scheduler.cancel(correlation_id))

    def _require_runner(self) -> ChatRunner:
        if self._runner is None:
            raise RuntimeError("AsyncChatEngine requires a ResumableChatRunner")
        return self._runner

    # ------------------------------------------------------------------
    # Internal loop helpers
    # ------------------------------------------------------------------

    async def _execute_start(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None,
        channel: EventChannel,
    ) -> ChatState:
        context = self._build_checkpoint_context(checkpoint=_IterationCheckpoint.START)
        return await self._execute_start_checkpoint(
            request, 0, 0, channel, hooks, context
        )

    async def _execute_before_iteration(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        payload: IterationCheckpointPayload,
        hooks: ExecutionHooks | None,
        channel: EventChannel,
    ) -> ChatState:
        context = self._build_checkpoint_context(
            checkpoint=_IterationCheckpoint.BEFORE_ITERATION,
            payload=payload,
        )
        return await self._execute_before_iteration_checkpoint(
            request, iteration, next_block_count, channel, hooks, context
        )

    async def _execute_close(
        self,
        request: ChatRequest,
        stop_reason: str | None,
        payload: IterationCheckpointPayload,
        hooks: ExecutionHooks | None,
        channel: EventChannel,
    ) -> ChatState:
        context = self._build_checkpoint_context(
            checkpoint=_IterationCheckpoint.CLOSE,
            stop_reason=stop_reason,
            payload=payload,
        )
        return await self._execute_close_checkpoint(
            request, 0, 0, channel, hooks, context
        )

    async def _continue_loop(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        payload: IterationCheckpointPayload,
        hooks: ExecutionHooks | None,
        channel: EventChannel,
    ) -> ChatState:
        while True:
            if iteration >= self._max_iterations:
                return await self._execute_close(
                    request, StopReasonEnum.MAX_TOKENS.value, payload, hooks, channel
                )
            state = await self._execute_before_iteration(
                request, iteration, next_block_count, payload, hooks, channel
            )
            new_payload = IterationCheckpointPayload(
                total_input_tokens=state.runtime.total_input_tokens,
                total_output_tokens=state.runtime.total_output_tokens,
                has_input_usage=state.runtime.has_input_usage,
                has_output_usage=state.runtime.has_output_usage,
            )
            if state.output.status == ChatStatus.WAITING:
                return state
            elif state.output.status == ChatStatus.COMPLETED:
                close_state = await self._execute_close(
                    state.input.request,
                    state.output.stop_reason,
                    new_payload,
                    hooks,
                    channel,
                )
                if state.output.pending_external_tool_calls:
                    close_state = close_state.model_copy(deep=True)
                    close_state.output.pending_external_tool_calls = list(
                        state.output.pending_external_tool_calls
                    )
                return close_state
            iteration = state.runtime.iteration
            next_block_count = state.runtime.next_block_count
            request = state.input.request
            payload = new_payload

    # ------------------------------------------------------------------
    # Phase producers
    # ------------------------------------------------------------------

    def _build_checkpoint_context(
        self,
        *,
        checkpoint: str,
        stop_reason: str | None = None,
        payload: IterationCheckpointPayload | None = None,
        context_stack: ContextStack | None = None,
    ) -> _CheckpointContext:
        return _CheckpointContext(
            checkpoint=checkpoint,
            stop_reason=stop_reason,
            context_stack=context_stack,
            payload=payload or IterationCheckpointPayload(),
        )

    def _resolve_checkpoint_handler(
        self, checkpoint: str
    ) -> Callable[
        [
            ChatRequest,
            int,
            int,
            EventChannel,
            ExecutionHooks | None,
            _CheckpointContext,
        ],
        Awaitable[ChatState],
    ]:
        if checkpoint not in self._checkpoint_handlers:
            raise ValueError(f"Unknown checkpoint: {checkpoint!r}")
        return cast(
            Callable[
                [
                    ChatRequest,
                    int,
                    int,
                    EventChannel,
                    ExecutionHooks | None,
                    _CheckpointContext,
                ],
                Awaitable[ChatState],
            ],
            self._checkpoint_handlers[checkpoint],
        )

    async def _execute_before_iteration_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        hooks: ExecutionHooks | None,
        checkpoint_context: _CheckpointContext,
    ) -> ChatState:
        run = self._initialize_run(
            request, context_stack=checkpoint_context.context_stack, hooks=hooks
        )
        run.state.runtime.iteration = iteration
        run.state.runtime.next_block_count = next_block_count
        run.block_count = next_block_count
        self._apply_payload_usage(run, checkpoint_context.payload)
        await self._run_intercepted_iteration(run, channel)
        self._store_runtime_usage(run)
        run.state = run.state.model_copy(deep=True)
        run.state.runtime.next_block_count = run.block_count
        return run.state.model_copy(deep=True)

    async def _execute_start_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        hooks: ExecutionHooks | None,
        checkpoint_context: _CheckpointContext,
    ) -> ChatState:
        del iteration, next_block_count
        run = self._initialize_run(
            request, context_stack=checkpoint_context.context_stack, hooks=hooks
        )
        channel.emit(RawMessageStartEvent.from_defaults())
        run.state = self._snapshot(run.state, TimelinePhase.START)
        run.state = run.state.model_copy(deep=True)
        await self.run_interceptor_phase(
            run,
            InterceptorPhase.VALIDATION,
            self._request_interceptors,
            channel,
        )
        run.state = run.state.model_copy(deep=True)
        run.state.output.status = ChatStatus.CONTINUE
        return run.state

    async def _process_tools_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        hooks: ExecutionHooks | None,
        checkpoint_context: _CheckpointContext,
    ) -> ChatState:
        return await self._continue_tools_checkpoint(
            request,
            iteration,
            next_block_count,
            channel,
            context_stack=checkpoint_context.context_stack,
            hooks=hooks,
            checkpoint_payload=checkpoint_context.payload,
        )

    async def _continue_tools_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        context_stack: ContextStack | None = None,
        hooks: ExecutionHooks | None = None,
        checkpoint_payload: IterationCheckpointPayload | None = None,
    ) -> ChatState:
        """Continue from the tools checkpoint without re-running the LLM call."""
        try:
            run = self._initialize_run(
                request, context_stack=context_stack, hooks=hooks
            )
            run.state.runtime.iteration = iteration
            run.state.runtime.next_block_count = next_block_count
            run.block_count = next_block_count
            payload = checkpoint_payload or IterationCheckpointPayload()
            self._apply_payload_usage(run, payload)
            pending_external = payload.pending_external_tool_calls

            for response in payload.tool_responses:
                result_start = RawContentBlockStartEvent(
                    index=run.block_count,
                    block_id=f"block_{uuid4().hex}",
                    content_block=ToolResultBlock(
                        tool_use_id=response.tool_id,
                        content=response.result_content,
                        is_error=response.is_error,
                    ),
                )
                run.block_count += 1
                channel.emit(result_start)
                channel.emit(RawContentBlockStopEvent.from_start(result_start))

            if pending_external:
                run.state = run.state.model_copy(deep=True)
                run.state.output.pending_external_tool_calls = pending_external
                run.state.output.stop_reason = StopReasonEnum.TOOL_USE.value
                run.state.output.status = ChatStatus.COMPLETED
                run.state = self._snapshot(run.state, TimelinePhase.STOP)
                return run.state

            run.state = self._snapshot(run.state, TimelinePhase.AFTER_TOOLS)
            return await self._execute_after_iteration_checkpoint(
                request,
                iteration,
                next_block_count,
                channel,
                hooks,
                self._build_checkpoint_context(
                    checkpoint=_IterationCheckpoint.AFTER_ITERATION,
                ),
                run=run,
            )
        except asyncio.CancelledError:
            logger.debug("tools checkpoint continuation cancelled")
            raise

    async def _execute_after_iteration_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        hooks: ExecutionHooks | None,
        checkpoint_context: _CheckpointContext,
        run: _Run | None = None,
    ) -> ChatState:
        del request, iteration, next_block_count, checkpoint_context
        if run is None:
            raise ValueError("after_iteration checkpoint requires prepared run state")
        await self.run_interceptor_phase(
            run,
            InterceptorPhase.AFTER_ITERATION,
            self._request_interceptors,
            channel,
        )
        run.state = run.state.model_copy(deep=True)
        run.state.runtime.next_block_count = run.block_count
        run.state.output.status = ChatStatus.CONTINUE
        return run.state

    async def _execute_close_checkpoint(
        self,
        request: ChatRequest,
        iteration: int,
        next_block_count: int,
        channel: EventChannel,
        hooks: ExecutionHooks | None,
        checkpoint_context: _CheckpointContext,
    ) -> ChatState:
        del iteration, next_block_count
        run = self._initialize_run(
            request, context_stack=checkpoint_context.context_stack, hooks=hooks
        )
        self._apply_payload_usage(run, checkpoint_context.payload)
        stop_reason = checkpoint_context.stop_reason
        run.state = run.state.model_copy(deep=True)
        run.state.output.stop_reason = stop_reason
        channel.emit(
            RawMessageDeltaEvent(
                delta=MessageOutputDelta(
                    stop_reason=stop_reason,
                    container=self._build_container(run),
                ),
                usage=self._build_total_usage(run),
            )
        )
        channel.emit(RawMessageStopEvent.from_defaults())
        run.state.output.status = ChatStatus.COMPLETED
        run.state = self._snapshot(run.state, TimelinePhase.STOP)
        return run.state

    # ------------------------------------------------------------------
    # Iteration machinery
    # ------------------------------------------------------------------

    def initialize_run(
        self,
        request: ChatRequest,
        context_stack: ContextStack | None = None,
        hooks: ExecutionHooks | None = None,
    ) -> _Run:
        return self._initialize_run(request, context_stack=context_stack, hooks=hooks)

    async def _run_intercepted_iteration(
        self,
        run: _Run,
        channel: EventChannel,
    ) -> None:
        iter_handler = _EventHandler(queue=asyncio.Queue())

        def current_context() -> ChatInterceptorContext:
            return ChatInterceptorContext(
                state=run.state,
                llm=run.llm,
                phase=InterceptorPhase.STREAMING,
                emit_fn=iter_handler.emit,
            )

        async def _run_and_close() -> None:
            try:
                for interceptor in self._response_interceptors:
                    await interceptor.on_iteration_start(current_context())
                await self._run_iteration(run, iter_handler)
            finally:
                for interceptor in self._response_interceptors:
                    with suppress(Exception):
                        await interceptor.on_iteration_end(current_context())
                await iter_handler.close()

        iter_task = asyncio.create_task(_run_and_close())

        async for raw_event in iter_handler.stream(iter_task):
            processed: Event | None = raw_event
            if processed is None:
                continue
            for interceptor in self._response_interceptors:
                processed = await interceptor.intercept_event(
                    processed, current_context()
                )
                if processed is None:
                    break
            if processed is not None:
                channel.emit(processed)

    async def _run_iteration(self, run: _Run, handler: _EventHandler) -> None:
        """One LLM call. Sets status; stop events are emitted by run_close."""
        run.state = run.state.model_copy(deep=True)

        await self.run_interceptor_phase(
            run,
            InterceptorPhase.BEFORE_ITERATION,
            self._request_interceptors,
            handler,
        )

        run.state.input.request = build_request_from_context_stack(
            run.state.input.request,
            run.state.input.context_stack,
        )
        run.state.runtime.iteration += 1
        run.state = self._snapshot(run.state, TimelinePhase.BEFORE_LLM)

        llm_tools = self._build_tools(run.state)
        tool_specs_by_name = {
            tool.name: tool
            for tool in run.state.input.context_stack.all_tools()
            if tool.name
        }
        schema_by_name: dict[str, dict[str, Any]] = {
            tool.name: tool.input_schema or {}
            for tool in run.state.input.context_stack.all_tools()
            if tool.name
        }
        llm_response = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT))

        lock = asyncio.Lock()
        stream_delta_state = _StreamDeltaState()
        if not run.state.input.request.tool_config.allow_parallel_tool_calls:
            stream_delta_state.tool_state.tool_semaphore = asyncio.Semaphore(1)

        llm_kwargs = run.state.input.llm_kwargs.copy()
        if isinstance(run.llm, ZylonLLM):
            llm_kwargs["priority"] = (
                run.state.input.request.system.priority
                or DefinedPriorities.LLM.CHAT_PRIORITY,
            )

        response_stream = await run.llm.astream_chat_with_tools(
            llm_tools,
            chat_history=run.state.input.request.to_messages(),
            allow_parallel_tool_calls=run.state.input.request.tool_config.allow_parallel_tool_calls,
            **run.state.input.llm_kwargs,
        )
        async for chunk in response_stream:
            llm_response = await self._handle_stream_chunk(
                run=run,
                llm=run.llm,
                chunk=chunk,
                current_response=llm_response,
                stream_delta_state=stream_delta_state,
                handler=handler,
                tool_specs_by_name=tool_specs_by_name,
                schema_by_name=schema_by_name,
                lock=lock,
            )

        self._close_active_stream_blocks(
            run=run,
            stream_delta_state=stream_delta_state,
            handler=handler,
            tool_specs_by_name=tool_specs_by_name,
            schema_by_name=schema_by_name,
            lock=lock,
        )

        for key, value in llm_response.additional_kwargs.items():
            llm_response.message.additional_kwargs.setdefault(key, value)

        tool_calls = await asyncio.to_thread(
            partial(
                run.llm.get_tool_calls_from_response,
                response=llm_response,
                error_on_no_tool_call=False,
            )
        )
        if tool_calls:
            llm_response.message.additional_kwargs["tool_calls"] = tool_calls
        else:
            llm_response.message.additional_kwargs.pop("tool_calls", None)

        assistant_message = llm_response.message
        self._ensure_update_tool_ids_in_tool_selection(
            run, assistant_message, stream_delta_state.tool_state.tool_id_map
        )
        self._validate_unique_tool_call_ids(assistant_message)
        self._accumulate_usage(run, assistant_message)

        run.state = run.state.model_copy(deep=True)
        run.state.input.request.messages = [
            *run.state.input.request.messages,
            assistant_message,
        ]
        run.state = self._snapshot(run.state, TimelinePhase.AFTER_LLM)

        if not tool_calls:
            stop_reason = assistant_message.additional_kwargs.get("stop_reason")
            run.state = run.state.model_copy(deep=True)
            run.state.output.stop_reason = stop_reason
            # COMPLETED: job dispatches close_chat_job — it emits stop events
            run.state.output.status = ChatStatus.COMPLETED
            return

        tool_call_results = await asyncio.gather(
            *stream_delta_state.tool_state.pending_tasks, return_exceptions=True
        )

        has_external_tool = False
        has_pending_tool = False
        pending_external = list(run.state.output.pending_external_tool_calls)
        pending_async = dict(run.state.output.pending_async_tools)
        for result in tool_call_results:
            if isinstance(result, Exception):
                raise result
            elif isinstance(result, _ToolExecutionResult):
                if result.status == _ToolExecutionStatus.NOT_EXECUTED:
                    has_external_tool = True
                    assert result.tool_selection is not None
                    pending_external.append(result.tool_selection)
                elif result.status == _ToolExecutionStatus.PENDING:
                    has_pending_tool = True
                    assert result.tool_selection is not None
                    tool_id = result.tool_selection.tool_id
                    if tool_id in pending_async:
                        raise RuntimeError(
                            f"Duplicate internal tool call ID '{tool_id}'"
                        )
                    pending_async[tool_id] = result.async_handle or ""

        if has_pending_tool:
            run.state = run.state.model_copy(deep=True)
            run.state.output.status = ChatStatus.WAITING
            run.state.output.pause_type = _IterationCheckpoint.TOOLS
            run.state.output.pending_async_tools = pending_async
            run.state.output.pending_external_tool_calls = pending_external
            run.state = self._snapshot(run.state, TimelinePhase.STOP)
            return

        if has_external_tool:
            run.state = run.state.model_copy(deep=True)
            run.state.output.pending_external_tool_calls = pending_external
            run.state.output.stop_reason = StopReasonEnum.TOOL_USE.value
            # COMPLETED: job dispatches close_chat_job
            run.state.output.status = ChatStatus.COMPLETED
            run.state = self._snapshot(run.state, TimelinePhase.STOP)
            return

        # Sync tools ran inline — run AFTER_TOOLS + AFTER_ITERATION, then CONTINUE
        run.state = self._snapshot(run.state, TimelinePhase.AFTER_TOOLS)

        await self.run_interceptor_phase(
            run,
            InterceptorPhase.AFTER_ITERATION,
            self._request_interceptors,
            handler,
        )

        run.state = run.state.model_copy(deep=True)
        run.state.output.status = ChatStatus.CONTINUE

    # ------------------------------------------------------------------
    # Stream chunk processing (identical logic to ChatLoopEngine)
    # ------------------------------------------------------------------

    async def _handle_stream_chunk(
        self,
        run: _Run,
        llm: FunctionCallingLLM,
        chunk: ChatResponse,
        current_response: ChatResponse,
        stream_delta_state: _StreamDeltaState,
        handler: _EventHandler,
        tool_specs_by_name: dict[str, ToolSpec],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> ChatResponse:
        assistant_message = current_response.message.model_copy(deep=True)
        if chunk.delta:
            assistant_message.content = (assistant_message.content or "") + chunk.delta
        elif (
            assistant_message.content is None
            and isinstance(chunk.message.content, str)
            and chunk.message.content
        ):
            assistant_message.content = chunk.message.content

        assistant_message.role = chunk.message.role or assistant_message.role
        for source in (chunk.additional_kwargs, chunk.message.additional_kwargs):
            for key, value in source.items():
                if key == "thinking_delta" and isinstance(value, str):
                    previous = assistant_message.additional_kwargs.get("thinking")
                    assistant_message.additional_kwargs["thinking"] = (
                        previous if isinstance(previous, str) else ""
                    ) + value
                    continue
                if key == "token_ids_delta" and isinstance(value, list):
                    previous = assistant_message.additional_kwargs.get(
                        "token_ids_delta"
                    )
                    existing = previous if isinstance(previous, list) else []
                    if not (
                        value
                        and len(existing) >= len(value)
                        and existing[-len(value) :] == value
                    ):
                        assistant_message.additional_kwargs["token_ids_delta"] = (
                            existing + value
                        )
                    continue
                if key == "tool_calls":
                    if isinstance(value, list) and value:
                        assistant_message.additional_kwargs["tool_calls"] = value
                    continue
                assistant_message.additional_kwargs[key] = value

        folded_response = ChatResponse(
            message=assistant_message,
            additional_kwargs=assistant_message.additional_kwargs,
        )

        tool_calls = await asyncio.to_thread(
            partial(
                llm.get_tool_calls_from_response,
                response=folded_response,
                error_on_no_tool_call=False,
            )
        )
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.tool_id is None:
                    continue

                raw_id = tool_call.tool_id
                tool_state = stream_delta_state.tool_state

                if raw_id not in tool_state.tool_id_map:
                    tool_state.tool_id_map[raw_id] = f"tool_{uuid4().hex}"

                if raw_id in tool_state.finished_tool_raw_ids:
                    continue

                if tool_state.active_tool_raw_id != raw_id:
                    if tool_state.active_tool_block is not None:
                        self._close_active_tool_block(
                            run=run,
                            stream_delta_state=stream_delta_state,
                            handler=handler,
                            tool_specs_by_name=tool_specs_by_name,
                            schema_by_name=schema_by_name,
                            lock=lock,
                        )

                    self._close_active_block(stream_delta_state, handler)

                    async with lock:
                        unique_id = tool_state.tool_id_map[raw_id]
                        use_start = RawContentBlockStartEvent(
                            index=run.block_count,
                            block_id=f"block_{uuid4().hex}",
                            content_block=ToolUseBlock(
                                id=unique_id,
                                name=tool_call.tool_name,
                                input={},
                            ),
                        )
                        run.block_count += 1
                        handler.emit(use_start)

                    tool_state.active_tool_block = use_start
                    tool_state.active_tool_raw_id = raw_id
                    tool_state.last_serialized[raw_id] = ""

                if tool_call.tool_kwargs and tool_state.active_tool_block is not None:
                    tool_schema = schema_by_name.get(tool_call.tool_name or "", {})
                    coerced_kwargs = (
                        _coerce_kwargs(tool_call.tool_kwargs, tool_schema)
                        if tool_schema
                        else tool_call.tool_kwargs
                    )
                    current_json = json.dumps(coerced_kwargs)
                    tool_state.last_serialized[raw_id] = current_json
                    handler.emit(
                        RawContentBlockDeltaEvent.from_content_block_start(
                            tool_state.active_tool_block,
                            InputJSONDelta(partial_json_obj=coerced_kwargs),
                        )
                    )
            return folded_response

        reasoning = self._extract_reasoning(chunk.message)
        if reasoning:
            self._switch_active_block(
                run=run,
                stream_delta_state=stream_delta_state,
                handler=handler,
                target_kind="thinking",
            )
            if stream_delta_state.active_block is not None:
                handler.emit(
                    RawContentBlockDeltaEvent.from_content_block_start(
                        stream_delta_state.active_block,
                        ThinkingDelta.from_text(reasoning),
                    )
                )
            return folded_response

        if chunk.delta:
            self._switch_active_block(
                run=run,
                stream_delta_state=stream_delta_state,
                handler=handler,
                target_kind="text",
            )
            if stream_delta_state.active_block is not None:
                handler.emit(
                    RawContentBlockDeltaEvent.from_content_block_start(
                        stream_delta_state.active_block,
                        TextDelta(text=chunk.delta),
                    )
                )
            return folded_response

        self._close_active_block(stream_delta_state, handler)
        return folded_response

    def _close_active_stream_blocks(
        self,
        run: _Run,
        stream_delta_state: _StreamDeltaState,
        handler: _EventHandler,
        tool_specs_by_name: dict[str, ToolSpec],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> None:
        if stream_delta_state.tool_state.active_tool_block is not None:
            self._close_active_tool_block(
                run,
                stream_delta_state,
                handler,
                tool_specs_by_name,
                schema_by_name,
                lock,
            )
        self._close_active_block(stream_delta_state, handler)

    def _close_active_tool_block(
        self,
        run: _Run,
        stream_delta_state: _StreamDeltaState,
        handler: _EventHandler,
        tool_specs_by_name: dict[str, ToolSpec],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> None:
        tool_state = stream_delta_state.tool_state
        if tool_state.active_tool_block is None:
            return

        prev_raw_id = tool_state.active_tool_raw_id or ""
        final_json = tool_state.last_serialized.get(prev_raw_id, "")
        final_obj: Any = json.loads(final_json) if final_json else {}

        tool_name = getattr(tool_state.active_tool_block.content_block, "name", None)
        if not isinstance(tool_name, str):
            raise TypeError("Active tool block must define a name")
        tool_schema = schema_by_name.get(tool_name, {})
        if tool_schema:
            final_obj = _coerce_kwargs(final_obj, tool_schema)

        handler.emit(
            RawContentBlockDeltaEvent.from_content_block_start(
                tool_state.active_tool_block,
                InputJSONDelta(
                    partial_json=final_json,
                    partial_json_obj=final_obj,
                ),
            )
        )
        handler.emit(RawContentBlockStopEvent.from_start(tool_state.active_tool_block))
        tool_state.finished_tool_raw_ids.add(prev_raw_id)

        tool_call = ToolSelection(
            tool_id=prev_raw_id,
            tool_name=tool_name,
            tool_kwargs=final_obj,
        )
        task = asyncio.create_task(
            self._handle_tool_use(
                run=run,
                tool_call=tool_call,
                tool_specs_by_name=tool_specs_by_name,
                handler=handler,
                tool_id_map=tool_state.tool_id_map,
                lock=lock,
                semaphore=stream_delta_state.tool_state.tool_semaphore,
            )
        )
        tool_state.pending_tasks.append(task)
        tool_state.active_tool_block = None
        tool_state.active_tool_raw_id = None

    @staticmethod
    def _close_active_block(
        stream_delta_state: _StreamDeltaState,
        handler: _EventHandler,
    ) -> None:
        if stream_delta_state.active_block is None:
            return
        handler.emit(
            RawContentBlockStopEvent.from_start(stream_delta_state.active_block)
        )
        stream_delta_state.active_block = None
        stream_delta_state.active_block_kind = None

    @staticmethod
    def _switch_active_block(
        run: _Run,
        stream_delta_state: _StreamDeltaState,
        handler: _EventHandler,
        target_kind: Literal["text", "thinking"],
    ) -> None:
        if stream_delta_state.active_block_kind == target_kind:
            return

        AsyncChatEngine._close_active_block(stream_delta_state, handler)
        if target_kind == "text":
            stream_delta_state.active_block = RawContentBlockStartEvent.from_text()
        else:
            stream_delta_state.active_block = RawContentBlockStartEvent(
                block_id=f"block_{uuid4().hex}",
                content_block=ThinkingBlock(
                    thinking="",
                    signature=f"sig_{uuid4().hex}",
                ),
            )
        stream_delta_state.active_block.index = run.block_count
        run.block_count += 1
        stream_delta_state.active_block_kind = target_kind
        handler.emit(stream_delta_state.active_block)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _handle_tool_use(
        self,
        run: _Run,
        tool_call: ToolSelection,
        tool_specs_by_name: dict[str, ToolSpec],
        handler: _EventHandler,
        tool_id_map: dict[str, str],
        lock: asyncio.Lock,
        semaphore: asyncio.Semaphore | None = None,
    ) -> _ToolExecutionResult:
        raw_id = tool_call.tool_id or ""
        call_id = tool_id_map.get(raw_id) or ""
        if not call_id:
            raise RuntimeError(
                f"Missing tool ID mapping for '{tool_call.tool_name}' with raw ID '{raw_id}'"
            )

        tool_spec = tool_specs_by_name.get(tool_call.tool_name or "")

        if tool_spec is None:
            error_content = f"Tool '{tool_call.tool_name}' not found."
            async with lock:
                error_start = RawContentBlockStartEvent(
                    index=run.block_count,
                    block_id=f"block_{uuid4().hex}",
                    content_block=ToolResultBlock(
                        tool_use_id=call_id,
                        content=error_content,
                        is_error=True,
                    ),
                )
                run.block_count += 1
                handler.emit(error_start)
                handler.emit(RawContentBlockStopEvent.from_start(error_start))

                error_message = ChatMessage(
                    role="tool",
                    content=error_content,
                    additional_kwargs={
                        "tool_call_id": call_id,
                        "tool_call_name": tool_call.tool_name,
                        "tool_call_args": tool_call.tool_kwargs,
                        "raw_output": error_content,
                    },
                )
                run.state = run.state.model_copy(deep=True)
                run.state.input.request.messages = [
                    *run.state.input.request.messages,
                    error_message,
                ]

            return _ToolExecutionResult(status=_ToolExecutionStatus.NOT_FOUND)

        if tool_spec.runtime != "server":
            return _ToolExecutionResult(
                status=_ToolExecutionStatus.NOT_EXECUTED,
                tool_selection=ToolSelection(
                    tool_id=call_id,
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                ),
            )

        if self._tool_scheduler.is_async:
            handle = await self._tool_scheduler.async_execute(
                ToolExecutionRequest(
                    tool_id=call_id,
                    tool_name=tool_call.tool_name or "",
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_spec=tool_spec,
                    context=build_tool_execution_context(run.state),
                    hooks=run.hooks,
                ),
                state_ctx=run.state,
                interceptors=self._tool_interceptors,
            )
            return _ToolExecutionResult(
                status=_ToolExecutionStatus.PENDING,
                tool_selection=ToolSelection(
                    tool_id=call_id,
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                ),
                async_handle=handle,
            )

        async with semaphore if semaphore is not None else contextlib.nullcontext():
            response = await self._tool_scheduler.execute(
                ToolExecutionRequest(
                    tool_id=call_id,
                    tool_name=tool_call.tool_name or "",
                    tool_kwargs=tool_call.tool_kwargs,
                    tool_spec=tool_spec,
                    context=build_tool_execution_context(run.state),
                    hooks=run.hooks,
                ),
                state_ctx=run.state,
                interceptors=self._tool_interceptors,
            )

        async with lock:
            run.state = run.state.model_copy(deep=True)
            run.state.input.request.messages = [
                *run.state.input.request.messages,
                response.tool_message,
            ]

        result_content = response.result_content

        async with lock:
            result_start = RawContentBlockStartEvent(
                index=run.block_count,
                block_id=f"block_{uuid4().hex}",
                content_block=ToolResultBlock(
                    tool_use_id=call_id,
                    content=result_content,
                    is_error=response.is_error,
                ),
            )
            run.block_count += 1
            handler.emit(result_start)
            handler.emit(RawContentBlockStopEvent.from_start(result_start))

        return _ToolExecutionResult(status=_ToolExecutionStatus.EXECUTED)

    # ------------------------------------------------------------------
    # Initialization and interceptor phases
    # ------------------------------------------------------------------

    def _initialize_run(
        self,
        request: ChatRequest,
        context_stack: ContextStack | None = None,
        hooks: ExecutionHooks | None = None,
    ) -> _Run:
        llm = self._llm_component.get_llm(request.system.model)
        if not isinstance(llm, FunctionCallingLLM):
            raise ValueError("Configured model does not support function calling")

        if not isinstance(request, ResolvedChatRequest) and context_stack is None:
            raise ValueError("Configured context stack is required")

        llm_kwargs = dict(request.sampling_params)
        if request.thinking.enabled and request.thinking.type:
            llm_kwargs["reasoning_effort"] = ReasoningEffort.from_str(
                request.thinking.type
            )
        if request.response_format and request.response_format.output_cls:
            structured = StructuredOutputsParams.from_optional(
                output_cls=request.response_format.output_cls,
            )
            if structured is not None:
                llm_kwargs["structured_outputs"] = structured

        state = ChatState(
            input=ChatInputState(
                request=request,
                context_stack=context_stack or build_initial_context_stack(request),
                sampling_params=dict(request.sampling_params),
                llm_kwargs=llm_kwargs,
            ),
            runtime=ChatRuntimeState(
                iteration=0,
                max_iterations=self._max_iterations,
            ),
            output=ChatOutputState(),
            timeline=[],
        )
        state.original_input = state.input.model_copy(deep=True)
        run = _Run(
            state=state,
            llm=llm,
            hooks=hooks or ExecutionHooks(),
        )
        for message in request.messages:
            self._accumulate_usage(run, message)
        self._store_runtime_usage(run)
        return run

    def _apply_payload_usage(
        self, run: _Run, payload: IterationCheckpointPayload
    ) -> None:
        if payload.has_input_usage:
            run.total_input_tokens = payload.total_input_tokens
            run.has_input_usage = True
        if payload.has_output_usage:
            run.total_output_tokens = payload.total_output_tokens
            run.has_output_usage = True
        self._store_runtime_usage(run)

    def _store_runtime_usage(self, run: _Run) -> None:
        run.state.runtime.total_input_tokens = run.total_input_tokens
        run.state.runtime.total_output_tokens = run.total_output_tokens
        run.state.runtime.has_input_usage = run.has_input_usage
        run.state.runtime.has_output_usage = run.has_output_usage

    async def run_interceptor_phase(
        self,
        run: _Run,
        phase: InterceptorPhase,
        interceptors: list[ChatRequestLoopInterceptor],
        channel: EventChannel | None = None,
    ) -> None:
        if not interceptors:
            return

        phase_marker = (
            TimelinePhase.BEFORE_INTERCEPTORS
            if phase == InterceptorPhase.BEFORE_ITERATION
            else TimelinePhase.AFTER_INTERCEPTORS
        )
        run.state = self._snapshot(run.state, phase_marker)

        for interceptor in interceptors:
            context = ChatInterceptorContext(
                state=run.state,
                llm=run.llm,
                phase=phase,
                emit_fn=channel.emit if channel is not None else lambda _event: None,
            )
            await interceptor.intercept(context)
            run.state = context.state

    def _build_tools(self, state: ChatState) -> list[AsyncBaseTool]:
        tool_specs = state.input.context_stack.all_tools()
        if not tool_specs:
            return []

        allowed_names = set(
            select_tool_names(
                tool_choices=state.input.request.tool_config.tool_choices,
                tool_names=[tool.name or "" for tool in tool_specs],
            )
        )
        selected = [tool for tool in tool_specs if (tool.name or "") in allowed_names]
        return [adapt_to_async_tool(tool.to_function_tool()) for tool in selected]

    # ------------------------------------------------------------------
    # Utility helpers (identical to ChatLoopEngine)
    # ------------------------------------------------------------------

    def _snapshot(self, state: ChatState, phase: TimelinePhase) -> ChatState:
        new_state = state.model_copy(deep=True)
        new_state.timeline.append(
            ChatTimelineEntry(
                iteration=new_state.runtime.iteration,
                phase=phase,
                conversation_size=len(new_state.input.request.to_messages()),
                tool_count=len(new_state.input.context_stack.all_tools()),
                stop_reason=new_state.output.stop_reason,
            )
        )
        return new_state

    @staticmethod
    def _extract_reasoning(message: ChatMessage) -> str | None:
        if message.additional_kwargs.get("stop_reason") is not None:
            return None
        raw_reasoning = message.additional_kwargs.get("thinking_delta")
        if isinstance(raw_reasoning, str):
            return raw_reasoning
        return None

    @staticmethod
    def _ensure_update_tool_ids_in_tool_selection(
        run: _Run, message: ChatMessage, tool_id_map: dict[str, str]
    ) -> None:
        tool_calls = message.additional_kwargs.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if isinstance(tool_call, ToolSelection):
                raw_id = tool_call.tool_id
                if raw_id and raw_id in tool_id_map:
                    tool_call.tool_id = tool_id_map[raw_id]

    @staticmethod
    def _validate_unique_tool_call_ids(message: ChatMessage) -> None:
        tool_calls = message.additional_kwargs.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return
        seen: set[str] = set()
        for tool_call in tool_calls:
            if not isinstance(tool_call, ToolSelection):
                continue
            tool_id = tool_call.tool_id
            if not tool_id:
                raise RuntimeError("Tool call is missing an ID")
            if tool_id in seen:
                raise RuntimeError(f"Duplicate tool call ID '{tool_id}'")
            seen.add(tool_id)

    @staticmethod
    def _accumulate_usage(run: _Run, message: ChatMessage) -> None:
        raw_input_tokens = message.additional_kwargs.get("input_tokens")
        if isinstance(raw_input_tokens, int):
            run.total_input_tokens += raw_input_tokens
            run.has_input_usage = True

        raw_output_tokens = message.additional_kwargs.get("output_tokens")
        if isinstance(raw_output_tokens, int):
            run.total_output_tokens += raw_output_tokens
            run.has_output_usage = True

    @staticmethod
    def _build_total_usage(run: _Run) -> Usage:
        return Usage(
            input_tokens=run.total_input_tokens if run.has_input_usage else None,
            output_tokens=run.total_output_tokens if run.has_output_usage else None,
        )

    def _build_container(self, run: _Run) -> Container | None:
        if self._container_registry is None:
            return None

        request = run.state.input.request
        if not isinstance(request, ResolvedChatRequest):
            return None

        session_id = _session_id(request)
        ttl = self._container_registry.get_ttl(session_id)
        if ttl is None:
            return None

        return Container(
            id=session_id,
            expires_at=datetime.now(tz=UTC) + timedelta(seconds=ttl),
        )
