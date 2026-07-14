import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
)
from private_gpt.components.engines.chat.chat_engine import ChatLoopEngine
from private_gpt.components.engines.chat.chat_runner import ChatRunner
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import ChatState
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.events.models import Event


@dataclass
class ChatEngineExecution:
    events: AsyncGenerator[Event, None]
    final_state_task: asyncio.Task[ChatState] | None
    execution_id: str | None = None


class ChatEngine(Protocol):
    async def run(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
        *,
        runner: ChatRunner | None = None,
    ) -> ChatEngineExecution:
        ...

    async def validate(self, request: ResolvedChatRequest) -> None:
        ...

    async def cancel(self, correlation_id: str) -> bool:
        ...


class LoopChatEngineAdapter:
    def __init__(self, engine: ChatLoopEngine) -> None:
        self._engine = engine

    async def run(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
        *,
        runner: ChatRunner | None = None,
    ) -> ChatEngineExecution:
        del runner
        execution = await self._engine.run(
            request=request,
            hooks=hooks.tool_result if hooks is not None else None,
        )
        return ChatEngineExecution(
            events=execution.events,
            final_state_task=execution.final_state_task,
        )

    async def validate(self, request: ResolvedChatRequest) -> None:
        await self._engine.run_interceptor_phase(
            run=self._engine.initialize_run(request),
            phase=InterceptorPhase.VALIDATION,
            interceptors=self._engine._request_interceptors,
        )

    async def cancel(self, correlation_id: str) -> bool:
        del correlation_id
        return True
