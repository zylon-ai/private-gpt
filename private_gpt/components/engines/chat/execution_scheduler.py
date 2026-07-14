from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from injector import Injector, inject, singleton

from private_gpt.settings.settings import Settings


class ChatExecutionScheduler(ABC):
    @abstractmethod
    async def start(
        self,
        *,
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    async def resume(self, *, execution_id: str) -> None:
        ...

    @abstractmethod
    async def callback(
        self, *, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> None:
        ...

    @abstractmethod
    async def timeout(
        self, *, execution_id: str, checkpoint_id: str, delay_seconds: int
    ) -> None:
        ...

    @abstractmethod
    async def cancel(self, execution_id: str) -> bool:
        ...


@singleton
class LocalChatExecutionScheduler(ChatExecutionScheduler):
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def _schedule(self, coro: Any, *, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def start(
        self,
        *,
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        from private_gpt.di import get_global_injector
        from private_gpt.server.chat.chat_service import ChatService

        engine = get_global_injector().get(ChatService).build_async_engine()
        self._schedule(
            engine.execute_scheduled_start(
                execution_id=execution_id,
                request_data=request_data,
                stream_type=stream_type,
                metadata=metadata,
            ),
            name=f"chat_{execution_id}",
        )

    async def resume(self, *, execution_id: str) -> None:
        from private_gpt.di import get_global_injector
        from private_gpt.server.chat.chat_service import ChatService

        engine = get_global_injector().get(ChatService).build_async_engine()
        self._schedule(
            engine.execute_scheduled_resume(execution_id=execution_id),
            name=f"chat_{execution_id}",
        )

    async def callback(
        self, *, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> None:
        from private_gpt.di import get_global_injector
        from private_gpt.server.chat.chat_service import ChatService

        chat_service = get_global_injector().get(ChatService)
        engine = chat_service.build_async_engine()
        await engine.record_callback(
            execution_id=execution_id,
            tool_id=tool_id,
            result=result,
        )

    async def timeout(
        self, *, execution_id: str, checkpoint_id: str, delay_seconds: int
    ) -> None:
        from private_gpt.di import get_global_injector
        from private_gpt.server.chat.chat_service import ChatService

        async def _timeout() -> None:
            await asyncio.sleep(delay_seconds)
            engine = get_global_injector().get(ChatService).build_async_engine()
            await engine.execute_timeout(
                execution_id=execution_id, checkpoint_id=checkpoint_id
            )

        self._schedule(_timeout(), name=f"chat_timeout_{execution_id}_{checkpoint_id}")

    async def cancel(self, execution_id: str) -> bool:
        cancelled = False
        for task in asyncio.all_tasks():
            if task.get_name().startswith(f"chat_{execution_id}"):
                task.cancel()
                cancelled = True
        return cancelled


@singleton
class ArqChatExecutionScheduler(ChatExecutionScheduler):
    async def start(
        self,
        *,
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        from private_gpt.arq.enqueue import enqueue_start_chat_job

        await enqueue_start_chat_job(
            request_data=request_data,
            correlation_id=execution_id,
            stream_type=stream_type,
            metadata=metadata,
            job_id=f"{execution_id}:start",
        )

    async def resume(self, *, execution_id: str) -> None:
        from private_gpt.arq.enqueue import enqueue_resume_iteration_job

        await enqueue_resume_iteration_job(
            correlation_id=execution_id,
            job_id=f"{execution_id}:resume",
        )

    async def callback(
        self, *, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> None:
        from private_gpt.arq.enqueue import enqueue_tool_resume_job

        await enqueue_tool_resume_job(
            correlation_id=execution_id,
            tool_id=tool_id,
            result=result,
        )

    async def timeout(
        self, *, execution_id: str, checkpoint_id: str, delay_seconds: int
    ) -> None:
        from private_gpt.arq.enqueue import enqueue_chat_timeout_job

        await enqueue_chat_timeout_job(
            correlation_id=execution_id,
            checkpoint_id=checkpoint_id,
            delay_seconds=delay_seconds,
            job_id=f"{execution_id}:timeout:{checkpoint_id}",
        )

    async def cancel(self, execution_id: str) -> bool:
        from private_gpt.arq.enqueue import abort_chat_job

        return await abort_chat_job(correlation_id=execution_id)


@singleton
class ChatExecutionSchedulerFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector

    def get(self) -> ChatExecutionScheduler:
        if self._settings.scheduler.chat.mode == "arq":
            return self._injector.get(ArqChatExecutionScheduler)
        return self._injector.get(LocalChatExecutionScheduler)
