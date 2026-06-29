"""Implement iterative chat loop execution without workflow memory."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from typing import Any, Literal
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import AsyncBaseTool, ToolSelection, adapt_to_async_tool

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
)
from private_gpt.components.container_registry import ContainerRegistry
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.interceptors.ensure_index_is_refreshed_interceptor import (
    EnsureIndexIsRefreshedInterceptor,
)
from private_gpt.components.engines.chat_loop.interceptors.ensure_timestamp_in_content_blocks_interceptors import (
    EnsureTimestampInContentBlocksInterceptor,
)
from private_gpt.components.engines.chat_loop.interceptors.ensure_tools_are_flatten_interceptor import (
    EnsureToolAreFlattenInterceptor,
)
from private_gpt.components.engines.chat_loop.interceptors.restore_stateless_input_interceptor import (
    RestoreStatelessInputInterceptorRequest,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
    TimelinePhase,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
    ChatLoopInputState,
    ChatLoopOutputState,
    ChatLoopRuntimeState,
    ChatLoopState,
    ChatLoopTimelineEntry,
)
from private_gpt.components.engines.chat_loop.utils.request_builder import (
    build_initial_context_stack,
    build_request_from_context_stack,
)
from private_gpt.components.engines.chat_loop.utils.tool_utils import (
    execute_tool_call,
    select_tool_names,
)
from private_gpt.components.llm.custom.base import StructuredOutputsParams, ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.llm.priorities import DefinedPriorities
from private_gpt.components.tools.processors.base import _session_id
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
    from_tool_output,
)
from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    _coerce_kwargs,
)

logger = logging.getLogger(__name__)


class ToolExecutionStatus(StrEnum):
    EXECUTED = "executed"
    NOT_EXECUTED = "not_executed"
    NOT_FOUND = "not_found"


@dataclass
class ToolExecutionResult:
    status: str
    tool_selection: ToolSelection | None = None


@dataclass
class _LoopRun:
    """Hold mutable runtime references for one run invocation."""

    state: ChatLoopState
    llm: FunctionCallingLLM
    stopped: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    has_input_usage: bool = False
    has_output_usage: bool = False
    block_count: int = 0


@dataclass
class _ToolDeltaState:
    """Track active tool streaming block and per-tool serialized input progress."""

    active_tool_block: RawContentBlockStartEvent | None = None
    active_tool_raw_id: str | None = None
    tool_id_map: dict[str, str] = field(default_factory=dict)
    finished_tool_raw_ids: set[str] = field(default_factory=set)
    last_serialized: dict[str, str] = field(default_factory=dict)

    pending_tasks: list[asyncio.Task[ToolExecutionResult]] = field(default_factory=list)
    tool_semaphore: asyncio.Semaphore | None = None


@dataclass
class _StreamDeltaState:
    """Track active streaming blocks while handling llm deltas."""

    active_block: RawContentBlockStartEvent | None = None
    active_block_kind: Literal["text", "thinking"] | None = None

    tool_state: _ToolDeltaState = field(default_factory=_ToolDeltaState)


class _LoopDone:
    """Mark producer completion in loop event queue."""


@dataclass
class _LoopEventHandler:
    """Provide real-time event emission and cancellation-safe streaming."""

    queue: asyncio.Queue[Event | _LoopDone]

    def emit(self, event: Event) -> None:
        """Emit one event immediately."""
        self.queue.put_nowait(event)

    def close(self) -> None:
        """Close event stream for the current loop."""
        self.queue.put_nowait(_LoopDone())

    async def stream(self, producer: asyncio.Task[None]) -> AsyncGenerator[Event, None]:
        """Stream queue events and cancel producer when consumer stops."""
        try:
            while True:
                queued = await self.queue.get()
                if isinstance(queued, _LoopDone):
                    break
                yield queued
        finally:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer


class ChatLoopEngine:
    """Run a modern iterative agent loop over chat messages and tools."""

    def __init__(
        self,
        llm_component: LLMComponent,
        request_interceptors: list[ChatRequestLoopInterceptor] | None = None,
        response_interceptors: list[ChatResponseLoopInterceptor] | None = None,
        max_iterations: int = 40,
        container_registry: ContainerRegistry | None = None,
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

    async def run(
        self,
        request: ChatRequest,
        context_stack: ContextStack | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Execute the loop and stream produced events immediately."""
        handler = _LoopEventHandler(queue=asyncio.Queue())
        producer = asyncio.create_task(self._run_loop(request, context_stack, handler))
        try:
            async for event in handler.stream(producer):
                yield event
        except asyncio.CancelledError:
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
            raise
        except:
            raise

    async def _run_loop(
        self,
        request: ChatRequest,
        context_stack: ContextStack | None,
        handler: _LoopEventHandler,
    ) -> None:
        try:
            run = self.initialize_run(request, context_stack)
            await self._run_loop_core(run, handler)
        except asyncio.CancelledError:
            logger.debug("Chat loop producer cancelled")
            raise
        finally:
            handler.close()

    async def _run_loop_core(self, run: _LoopRun, handler: _LoopEventHandler) -> None:
        handler.emit(RawMessageStartEvent.from_defaults())
        run.state = self._snapshot(run.state, TimelinePhase.START)

        run.state = run.state.model_copy(deep=True)
        await self.run_interceptor_phase(
            run,
            InterceptorPhase.VALIDATION,
            self._request_interceptors,
            handler,
        )

        while (
            not run.stopped
            and run.state.runtime.iteration < run.state.runtime.max_iterations
        ):
            await self._run_intercepted_iteration(run, handler)

        if not run.stopped:
            run.state = run.state.model_copy(deep=True)
            run.state.output.stop_reason = StopReasonEnum.MAX_TOKENS.value
            handler.emit(
                RawMessageDeltaEvent(
                    delta=MessageOutputDelta(
                        stop_reason=StopReasonEnum.MAX_TOKENS.value
                    ),
                    usage=self._build_total_usage(run),
                )
            )
            handler.emit(RawMessageStopEvent.from_defaults())
            run.state = self._snapshot(run.state, TimelinePhase.STOP)

    async def _run_intercepted_iteration(
        self,
        run: _LoopRun,
        outer_handler: _LoopEventHandler,
    ) -> None:
        iter_handler = _LoopEventHandler(queue=asyncio.Queue())

        def current_context() -> ChatLoopInterceptorContext:
            return ChatLoopInterceptorContext(
                state=run.state,
                llm=run.llm,
                phase=InterceptorPhase.STREAMING,
                emit_fn=iter_handler.emit,
            )

        async def _run_and_close() -> None:
            try:
                # Run iteration with before/after interceptors
                # and ensure iteration-end logic runs
                for interceptor in self._response_interceptors:
                    await interceptor.on_iteration_start(current_context())

                await self._run_safe_iteration(run, iter_handler)
            finally:
                # Ensure that all iteration-end logic runs and events are emitted
                # even if the iteration task is cancelled
                for interceptor in self._response_interceptors:
                    with suppress(Exception):
                        await interceptor.on_iteration_end(current_context())

                iter_handler.close()

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
                outer_handler.emit(processed)

    async def _run_safe_iteration(
        self, run: _LoopRun, handler: _LoopEventHandler
    ) -> None:
        """Run one iteration and catch exceptions to prevent producer cancellation."""
        try:
            await self._run_iteration(run, handler)

        except asyncio.CancelledError:
            logger.debug("Chat loop iteration cancelled")
            raise
        except Exception:
            raise

    async def _run_iteration(self, run: _LoopRun, handler: _LoopEventHandler) -> None:
        """Run one iteration with llm call, tool execution, and after interceptors."""
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
        tool_by_name = {tool.metadata.name: tool for tool in llm_tools}
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
                tool_by_name=tool_by_name,
                schema_by_name=schema_by_name,
                lock=lock,
            )

        self._close_active_stream_blocks(
            run=run,
            stream_delta_state=stream_delta_state,
            handler=handler,
            tool_by_name=tool_by_name,
            schema_by_name=schema_by_name,
            lock=lock,
        )

        # The order is important:
        # OpenAI returns a custom model instead of ToolSelection
        # We have to extract, and override before to add the history again
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
            run.stopped = True
            handler.emit(
                RawMessageDeltaEvent(
                    delta=MessageOutputDelta(
                        stop_reason=stop_reason,
                        container=self._build_container(run),
                    ),
                    usage=self._build_total_usage(run),
                )
            )
            handler.emit(RawMessageStopEvent.from_defaults())
            run.state = self._snapshot(run.state, TimelinePhase.STOP)
            return

        # Await all tool tasks spawned eagerly during streaming
        tool_call_results = await asyncio.gather(
            *stream_delta_state.tool_state.pending_tasks, return_exceptions=True
        )

        has_external_tool = False
        pending_external = list(run.state.output.pending_external_tool_calls)
        for result in tool_call_results:
            if isinstance(result, Exception):
                raise result
            elif isinstance(result, ToolExecutionResult):
                if result.status == ToolExecutionStatus.NOT_EXECUTED:
                    has_external_tool = True
                    assert result.tool_selection is not None
                    pending_external.append(result.tool_selection)

        if has_external_tool:
            run.state = run.state.model_copy(deep=True)
            run.state.output.pending_external_tool_calls = pending_external
            run.state.output.stop_reason = StopReasonEnum.TOOL_USE.value
            run.stopped = True
            handler.emit(
                RawMessageDeltaEvent(
                    delta=MessageOutputDelta(stop_reason=StopReasonEnum.TOOL_USE.value),
                    usage=self._build_total_usage(run),
                )
            )
            handler.emit(RawMessageStopEvent.from_defaults())
            run.state = self._snapshot(run.state, TimelinePhase.STOP)
            return

        run.state = self._snapshot(run.state, TimelinePhase.AFTER_TOOLS)

        await self.run_interceptor_phase(
            run,
            InterceptorPhase.AFTER_ITERATION,
            self._request_interceptors,
            handler,
        )

    async def _handle_stream_chunk(
        self,
        run: _LoopRun,
        llm: FunctionCallingLLM,
        chunk: ChatResponse,
        current_response: ChatResponse,
        stream_delta_state: _StreamDeltaState,
        handler: _LoopEventHandler,
        tool_by_name: dict[str | None, AsyncBaseTool],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> ChatResponse:
        """Handle one llm chunk with single-active-block transitions."""
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
                    # Close previous tool block with final serialized partial_json
                    if tool_state.active_tool_block is not None:
                        self._close_active_tool_block(
                            run=run,
                            stream_delta_state=stream_delta_state,
                            handler=handler,
                            tool_by_name=tool_by_name,
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

                # During streaming: emit partial_json_obj only, no partial_json string
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
        run: _LoopRun,
        stream_delta_state: _StreamDeltaState,
        handler: _LoopEventHandler,
        tool_by_name: dict[str | None, AsyncBaseTool],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> None:
        """Close all active streaming blocks and spawn the last tool task."""
        if stream_delta_state.tool_state.active_tool_block is not None:
            self._close_active_tool_block(
                run, stream_delta_state, handler, tool_by_name, schema_by_name, lock
            )
        self._close_active_block(stream_delta_state, handler)

    def _close_active_tool_block(
        self,
        run: _LoopRun,
        stream_delta_state: _StreamDeltaState,
        handler: _LoopEventHandler,
        tool_by_name: dict[str | None, AsyncBaseTool],
        schema_by_name: dict[str, dict[str, Any]],
        lock: asyncio.Lock,
    ) -> None:
        """Close the active tool block and spawn its execution task immediately."""
        tool_state = stream_delta_state.tool_state
        if tool_state.active_tool_block is None:
            return

        prev_raw_id = tool_state.active_tool_raw_id or ""
        final_json = tool_state.last_serialized.get(prev_raw_id, "")
        final_obj: Any = json.loads(final_json) if final_json else {}

        tool_name = tool_state.active_tool_block.content_block.name  # type: ignore[union-attr]
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
                tool_by_name=tool_by_name,
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
        handler: _LoopEventHandler,
    ) -> None:
        """Close the current active text/thinking block when present."""
        if stream_delta_state.active_block is None:
            return
        handler.emit(
            RawContentBlockStopEvent.from_start(stream_delta_state.active_block)
        )
        stream_delta_state.active_block = None
        stream_delta_state.active_block_kind = None

    @staticmethod
    def _switch_active_block(
        run: _LoopRun,
        stream_delta_state: _StreamDeltaState,
        handler: _LoopEventHandler,
        target_kind: Literal["text", "thinking"],
    ) -> None:
        """Ensure exactly one active block and switch by kind."""
        if stream_delta_state.active_block_kind == target_kind:
            return

        ChatLoopEngine._close_active_block(stream_delta_state, handler)
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

    def initialize_run(
        self,
        request: ChatRequest,
        context_stack: ContextStack | None = None,
    ) -> _LoopRun:
        """Build initial llm and state for one run."""
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

        state = ChatLoopState(
            input=ChatLoopInputState(
                request=request,
                context_stack=context_stack or build_initial_context_stack(request),
                sampling_params=dict(request.sampling_params),
                llm_kwargs=llm_kwargs,
            ),
            runtime=ChatLoopRuntimeState(
                iteration=0,
                max_iterations=self._max_iterations,
            ),
            output=ChatLoopOutputState(),
            timeline=[],
        )
        state.original_input = state.input.model_copy(deep=True)
        return _LoopRun(state=state, llm=llm)

    async def run_interceptor_phase(
        self,
        run: _LoopRun,
        phase: InterceptorPhase,
        interceptors: list[ChatRequestLoopInterceptor],
        handler: _LoopEventHandler | None = None,
    ) -> None:
        """Run one interceptor phase and emit events via shared handler."""
        if not interceptors:
            return

        phase_marker = (
            TimelinePhase.BEFORE_INTERCEPTORS
            if phase == InterceptorPhase.BEFORE_ITERATION
            else TimelinePhase.AFTER_INTERCEPTORS
        )
        run.state = self._snapshot(run.state, phase_marker)

        for interceptor in interceptors:
            context = ChatLoopInterceptorContext(
                state=run.state,
                llm=run.llm,
                phase=phase,
                emit_fn=handler.emit if handler is not None else lambda _event: None,
            )
            await interceptor.intercept(context)
            run.state = context.state

    def _build_tools(self, state: ChatLoopState) -> list[AsyncBaseTool]:
        """Resolve tool specs into async llama-index tools for current state."""
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

    async def _handle_tool_use(
        self,
        run: _LoopRun,
        tool_call: ToolSelection,
        tool_by_name: dict[str | None, AsyncBaseTool],
        handler: _LoopEventHandler,
        tool_id_map: dict[str, str],
        lock: asyncio.Lock,
        semaphore: asyncio.Semaphore | None = None,
    ) -> ToolExecutionResult:
        raw_id = tool_call.tool_id or ""
        call_id = tool_id_map.get(raw_id) or ""
        if not call_id:
            raise RuntimeError(
                f"Missing tool ID mapping for '{tool_call.tool_name}' with raw ID '{raw_id}'"
            )

        tool = tool_by_name.get(tool_call.tool_name)

        if tool is None:
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

            return ToolExecutionResult(status=ToolExecutionStatus.NOT_FOUND)

        if tool.metadata.return_direct:
            return ToolExecutionResult(
                status=ToolExecutionStatus.NOT_EXECUTED,
                tool_selection=ToolSelection(
                    tool_id=call_id,
                    tool_name=tool_call.tool_name,
                    tool_kwargs=tool_call.tool_kwargs,
                ),
            )

        async with semaphore if semaphore is not None else contextlib.nullcontext():
            result, tool_message = await execute_tool_call(
                tool=tool,
                tool_name=tool_call.tool_name,
                tool_id=call_id,
                tool_kwargs=tool_call.tool_kwargs,
                state_ctx=run.state,
            )

        async with lock:
            run.state = run.state.model_copy(deep=True)
            run.state.input.request.messages = [
                *run.state.input.request.messages,
                tool_message,
            ]

        result_content = (
            from_tool_output(result.tool_output.raw_output)
            if result.tool_output.raw_output is not None
            else result.tool_output.content
        )

        async with lock:
            result_start = RawContentBlockStartEvent(
                index=run.block_count,
                block_id=f"block_{uuid4().hex}",
                content_block=ToolResultBlock(
                    tool_use_id=call_id,
                    content=result_content,
                    is_error=result.tool_output.is_error,
                ),
            )

            run.block_count += 1
            handler.emit(result_start)
            handler.emit(RawContentBlockStopEvent.from_start(result_start))

        return ToolExecutionResult(status=ToolExecutionStatus.EXECUTED)

    def _snapshot(self, state: ChatLoopState, phase: TimelinePhase) -> ChatLoopState:
        """Append one immutable timeline entry."""
        new_state = state.model_copy(deep=True)
        new_state.timeline.append(
            ChatLoopTimelineEntry(
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
        """Extract reasoning text from one assistant message chunk."""
        if message.additional_kwargs.get("stop_reason") is not None:
            return None
        raw_reasoning = message.additional_kwargs.get("thinking_delta")
        if isinstance(raw_reasoning, str):
            return raw_reasoning
        return None

    @staticmethod
    def _ensure_update_tool_ids_in_tool_selection(
        run: _LoopRun, message: ChatMessage, tool_id_map: dict[str, str]
    ) -> None:
        """Ensure that all tool selections emitted."""
        tool_calls = message.additional_kwargs.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return

        for tool_call in tool_calls:
            if isinstance(tool_call, ToolSelection):
                raw_id = tool_call.tool_id
                if raw_id and raw_id in tool_id_map:
                    tool_call.tool_id = tool_id_map[raw_id]

    @staticmethod
    def _accumulate_usage(run: _LoopRun, message: ChatMessage) -> None:
        """Accumulate usage counters from one assistant message."""
        raw_input_tokens = message.additional_kwargs.get("input_tokens")
        if isinstance(raw_input_tokens, int):
            run.total_input_tokens += raw_input_tokens
            run.has_input_usage = True

        raw_output_tokens = message.additional_kwargs.get("output_tokens")
        if isinstance(raw_output_tokens, int):
            run.total_output_tokens += raw_output_tokens
            run.has_output_usage = True

    @staticmethod
    def _build_total_usage(run: _LoopRun) -> Usage:
        """Build cumulative usage from all completed iterations."""
        return Usage(
            input_tokens=run.total_input_tokens if run.has_input_usage else None,
            output_tokens=run.total_output_tokens if run.has_output_usage else None,
        )

    def _build_container(self, run: _LoopRun) -> Container | None:
        """Return a container handle if a persistent session was registered."""
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
