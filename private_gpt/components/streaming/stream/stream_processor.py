import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.stream.event_handler import EventHandler
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
        try:
            await self.stream_service.update_stream_status(
                correlation_id,
                StreamStatus.PENDING,
                metadata=metadata,
            )

            is_processing = False
            async for event in event_generator:
                if self.task_manager.is_cancelled(correlation_id):
                    raise asyncio.CancelledError(
                        f"Stream processing for {correlation_id} has been cancelled."
                    )

                # Also check the stream-service cancel flag so that cancellation
                # requests from the API process propagate to a chat worker
                # running in a separate Celery process.
                if await self.stream_service.is_cancelled(correlation_id):
                    raise asyncio.CancelledError(
                        f"Stream processing for {correlation_id} has been cancelled "
                        "via stream-service flag."
                    )

                if not is_processing:
                    # Do a lazy initialization of processing status.
                    current_status = await event_handler.get_current_status(event)
                    if (
                        current_status is not None
                        and current_status >= StreamStatus.PROCESSING
                    ):
                        # Don't submit current status since
                        # the current event has not been processed yet.
                        await self.stream_service.update_stream_status(
                            correlation_id,
                            StreamStatus.PROCESSING,
                        )
                        is_processing = True

                event_data = await asyncio.to_thread(event_handler.serialize, event)
                await self.stream_service.push_event(
                    correlation_id=correlation_id,
                    event_data=event_data,
                )

            await self.stream_service.update_stream_status(
                correlation_id,
                StreamStatus.COMPLETED,
            )
            await self.stream_service.clear_cancel_flag(correlation_id)

        except asyncio.CancelledError:
            await self.stream_service.update_stream_status(
                correlation_id,
                StreamStatus.CANCELLED,
            )
            await self.stream_service.clear_cancel_flag(correlation_id)
            await self._handle_cancellation(correlation_id, event_generator)
            raise
        except Exception as e:
            if not isinstance(e, asyncio.CancelledError):

                def create_error_event_data(exception: Exception) -> str:
                    error_evt = event_handler.error_event(correlation_id, exception)
                    return event_handler.serialize(error_evt)

                event_data = await asyncio.to_thread(create_error_event_data, e)
                await self.stream_service.push_event(
                    correlation_id=correlation_id,
                    event_data=event_data,
                )
                await self.stream_service.update_stream_status(
                    correlation_id,
                    StreamStatus.ERROR,
                    error_message=str(e),
                )
                await self.stream_service.clear_cancel_flag(correlation_id)
            raise

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
        task: asyncio.Task[Any] = await self.task_manager.create_task(
            correlation_id=correlation_id,
            coro=self.process_stream(
                correlation_id=correlation_id,
                stream_type=stream_type,
                event_generator=event_generator,
                event_handler=event_handler,
                metadata=metadata,
            ),
            name=f"stream_processor_{correlation_id}",
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
