import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.stream.event_handler import EventHandler
from private_gpt.components.streaming.stream.runtime import (
    process_stream as run_stream_processing,
)
from private_gpt.components.streaming.stream.runtime import (
    start_stream_processing as launch_stream_processing,
)
from private_gpt.components.streaming.stream_component import StreamComponent
from private_gpt.components.streaming.tasks.chat_scheduler import ChatSchedulerFactory
from private_gpt.components.streaming.tasks.task_manager import TaskManager


@singleton
class StreamProcessor:
    """Processes event streams and stores minimal data in Redis."""

    @inject
    def __init__(
        self,
        stream_component: StreamComponent,
        task_manager: TaskManager,
        chat_scheduler_factory: ChatSchedulerFactory,
    ):
        self.stream_service = stream_component.stream
        self.task_manager = task_manager
        self.chat_scheduler = chat_scheduler_factory.get()

    async def process_stream(
        self,
        correlation_id: str,
        stream_type: str,
        event_generator: AsyncGenerator[BaseModel, None],
        event_handler: EventHandler,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Process a stream of events and push to Redis."""
        await run_stream_processing(
            stream_service=self.stream_service,
            task_manager=self.task_manager,
            correlation_id=correlation_id,
            stream_type=stream_type,
            event_generator=event_generator,
            event_handler=event_handler,
            metadata=metadata,
        )

    async def _handle_cancellation(
        self,
        correlation_id: str,
        event_generator: AsyncGenerator[BaseModel, None],
    ) -> None:
        """Handle cancellation of the stream generator."""
        try:
            if hasattr(event_generator, "cancel"):
                await event_generator.cancel()
            elif hasattr(event_generator, "aclose"):
                await event_generator.aclose()
        except Exception:
            pass

    async def start_stream_processing(
        self,
        correlation_id: str,
        stream_type: str,
        event_generator: AsyncGenerator[BaseModel, None],
        event_handler: EventHandler,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        """Start processing a stream in the background."""
        task: asyncio.Task[Any] = await launch_stream_processing(
            stream_service=self.stream_service,
            task_manager=self.task_manager,
            correlation_id=correlation_id,
            stream_type=stream_type,
            event_generator=event_generator,
            event_handler=event_handler,
            metadata=metadata,
        )
        return task

    async def cancel_stream_processing(self, correlation_id: str) -> bool:
        """Cancel stream processing.

        Signals cancellation through two channels:
        1. The in-process ``TaskManager`` cancellation token + asyncio task
           cancel (works when the chat loop runs on the API event loop).
        2. The stream-service cancel flag (works when the chat loop runs in a
           separate Celery worker process that polls the flag).
        """
        await self.stream_service.set_cancel_flag(correlation_id)
        scheduled_cancel = await self.chat_scheduler.cancel(correlation_id)
        success = await self.task_manager.cancel_task(correlation_id)
        if success or scheduled_cancel:
            await self.stream_service.update_stream_status(
                correlation_id,
                StreamStatus.CANCELLED,
            )
        return success or scheduled_cancel
