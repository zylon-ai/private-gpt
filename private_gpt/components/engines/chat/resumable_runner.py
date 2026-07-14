from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from injector import inject, singleton
from llama_index.core.tools import ToolSelection

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.engines.chat.checkpoint_store import (
    ChatCheckpoint,
    ChatCheckpointStoreFactory,
)
from private_gpt.components.engines.chat.event_broker import EngineEventBrokerFactory
from private_gpt.components.engines.chat.event_channel import BrokerEventChannel
from private_gpt.components.engines.chat.execution_scheduler import (
    ChatExecutionSchedulerFactory,
)
from private_gpt.components.engines.chat.models.chat_state import ChatStatus
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
    ToolExecutionHook,
)
from private_gpt.components.tools.tool_scheduler import ToolSchedulerFactory
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
    from private_gpt.components.engines.chat.models.chat_state import ChatState
    from private_gpt.events.models import Event


_RESUME_HOOKS = ExecutionHooks(
    tool_result=[
        ToolExecutionHook(
            callable_path="private_gpt.arq.tasks.chat.callback:resume_chat_callback"
        )
    ]
)


@singleton
class ResumableChatRunner:
    """Single start/resume/timeout implementation for local and ARQ execution."""

    @inject
    def __init__(
        self,
        settings: Settings,
        checkpoint_store_factory: ChatCheckpointStoreFactory,
        event_broker_factory: EngineEventBrokerFactory,
        scheduler_factory: ChatExecutionSchedulerFactory,
        tool_scheduler_factory: ToolSchedulerFactory,
    ) -> None:
        self._settings = settings
        self._state = checkpoint_store_factory.get()
        self._events = event_broker_factory.get()
        self._scheduler = scheduler_factory.get()
        self._tool_scheduler = tool_scheduler_factory.get()

    async def submit(
        self,
        *,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
        execution_id: str | None = None,
    ) -> tuple[str, AsyncGenerator[Event, None]]:
        execution_id = execution_id or str(uuid4())
        events = self._events.listen(execution_id)
        await self._scheduler.start(
            execution_id=execution_id,
            request_data=request_data,
            stream_type=stream_type,
            metadata=metadata,
        )
        return execution_id, events

    async def cancel(self, execution_id: str) -> bool:
        checkpoint = await self._state.load(execution_id)
        tool_task_ids = (
            list(checkpoint.checkpoint_payload.pending_async_tools.values())
            if checkpoint is not None
            else []
        )
        tool_cancellations = await asyncio.gather(
            *(
                self._tool_scheduler.cancel_task(task_id=task_id)
                for task_id in tool_task_ids
            )
        )
        chat_cancelled = await self._scheduler.cancel(execution_id)
        await self._state.cleanup(execution_id)
        await self._events.finish(execution_id)
        return chat_cancelled or any(tool_cancellations)

    async def start(
        self,
        *,
        engine: AsyncChatEngine,
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        channel = BrokerEventChannel(self._events, execution_id)
        try:
            request = self._request(request_data)
            state = await engine.execute(request, hooks=_RESUME_HOOKS, channel=channel)
            await channel.close()
            await self._handle_state(
                execution_id=execution_id,
                state=state,
                stream_type=stream_type,
                metadata=metadata,
            )
        except Exception as exc:
            await self._fail(execution_id, exc, channel)
            raise

    async def resume(self, *, engine: AsyncChatEngine, execution_id: str) -> None:
        saved = await self._state.load(execution_id)
        if saved is None:
            return
        channel = BrokerEventChannel(self._events, execution_id)
        try:
            responses = list((await self._state.get_results(execution_id)).values())
            request_data = dict(saved.request_data)
            request_data["messages"] = [
                *list(request_data.get("messages", [])),
                *(
                    response.tool_message.model_dump(mode="json")
                    for response in responses
                ),
            ]
            state = await engine.resume(
                saved.checkpoint,
                self._request(request_data),
                iteration=saved.iteration,
                next_block_count=saved.next_block_count,
                hooks=_RESUME_HOOKS,
                checkpoint_payload=saved.checkpoint_payload.model_copy(
                    update={"tool_responses": responses}
                ),
                channel=channel,
            )
            await channel.close()
            await self._state.cleanup(execution_id)
            await self._handle_state(
                execution_id=execution_id,
                state=state,
                stream_type=saved.stream_type,
                metadata=saved.metadata,
            )
        except Exception as exc:
            await self._fail(execution_id, exc, channel)
            raise

    async def callback(
        self, *, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> None:
        ready = await self._state.record_result(execution_id, tool_id, result)
        if ready is None or not await self._state.claim_resume(execution_id):
            return
        await self._scheduler.resume(execution_id=execution_id)

    async def timeout(self, *, execution_id: str, checkpoint_id: str) -> None:
        saved = await self._state.load(execution_id)
        if saved is None or saved.checkpoint_id != checkpoint_id:
            return
        if not await self._state.claim_resume(execution_id):
            return
        await self._fail(
            execution_id,
            TimeoutError(
                f"Chat callback timed out after {self._settings.scheduler.chat.callback_timeout_seconds} seconds"
            ),
        )

    async def _handle_state(
        self,
        *,
        execution_id: str,
        state: ChatState,
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        if state.output.status != ChatStatus.WAITING:
            await self._state.cleanup(execution_id)
            await self._events.finish(execution_id)
            return

        checkpoint_id = uuid4().hex
        timeout_seconds = self._settings.scheduler.chat.callback_timeout_seconds
        await self._state.save(
            ChatCheckpoint(
                correlation_id=execution_id,
                request_data=state.input.request.model_dump(mode="json"),
                stream_type=stream_type,
                metadata=metadata,
                iteration=state.runtime.iteration,
                checkpoint=state.output.pause_type,
                checkpoint_payload=self._checkpoint_payload(state),
                next_block_count=state.runtime.next_block_count,
                checkpoint_id=checkpoint_id,
                deadline=datetime.now(UTC) + timedelta(seconds=timeout_seconds),
            )
        )
        await self._scheduler.timeout(
            execution_id=execution_id,
            checkpoint_id=checkpoint_id,
            delay_seconds=timeout_seconds,
        )

    async def _fail(
        self,
        execution_id: str,
        exc: Exception,
        channel: BrokerEventChannel | None = None,
    ) -> None:
        event = StreamingEventHandler().error_event(execution_id, exc)
        if channel is None:
            await self._events.publish(execution_id, event)
        else:
            channel.emit(event)
            await channel.close()
        await self._state.cleanup(execution_id)
        await self._events.finish(execution_id)

    @staticmethod
    def _request(request_data: dict[str, Any]) -> ResolvedChatRequest:
        request = ResolvedChatRequest.model_validate(request_data)
        for message in request.messages:
            tool_calls = message.additional_kwargs.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            message.additional_kwargs["tool_calls"] = [
                ToolSelection.model_validate(tool_call)
                if isinstance(tool_call, dict)
                else tool_call
                for tool_call in tool_calls
            ]
        return request

    @staticmethod
    def _checkpoint_payload(state: ChatState) -> Any:
        from private_gpt.components.engines.chat.async_chat_engine import (
            IterationCheckpointPayload,
        )

        return IterationCheckpointPayload(
            pending_async_tools=state.output.pending_async_tools,
            pending_external_tool_calls=state.output.pending_external_tool_calls,
            total_input_tokens=state.runtime.total_input_tokens,
            total_output_tokens=state.runtime.total_output_tokens,
            has_input_usage=state.runtime.has_input_usage,
            has_output_usage=state.runtime.has_output_usage,
        )
