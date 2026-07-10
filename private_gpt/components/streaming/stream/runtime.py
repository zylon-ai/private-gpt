import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.providers.stream_service import StreamService
from private_gpt.components.streaming.stream.event_handler import EventHandler
from private_gpt.components.streaming.tasks.task_manager import TaskManager


async def process_stream(
    *,
    stream_service: StreamService,
    task_manager: TaskManager,
    correlation_id: str,
    stream_type: str,
    event_generator: AsyncGenerator[BaseModel, None],
    event_handler: EventHandler,
    metadata: dict[str, Any] | None = None,
) -> None:
    del stream_type
    try:
        await stream_service.update_stream_status(
            correlation_id,
            StreamStatus.PENDING,
            metadata=metadata,
        )

        is_processing = False
        async for event in event_generator:
            if task_manager.is_cancelled(correlation_id):
                raise asyncio.CancelledError(
                    f"Stream processing for {correlation_id} has been cancelled."
                )

            if await stream_service.is_cancelled(correlation_id):
                raise asyncio.CancelledError(
                    f"Stream processing for {correlation_id} has been cancelled via stream-service flag."
                )

            if not is_processing:
                current_status = await event_handler.get_current_status(event)
                if (
                    current_status is not None
                    and current_status >= StreamStatus.PROCESSING
                ):
                    await stream_service.update_stream_status(
                        correlation_id,
                        StreamStatus.PROCESSING,
                    )
                    is_processing = True

            event_data = await asyncio.to_thread(event_handler.serialize, event)
            await stream_service.push_event(
                correlation_id=correlation_id,
                event_data=event_data,
            )

        await stream_service.update_stream_status(
            correlation_id,
            StreamStatus.COMPLETED,
        )
        await stream_service.clear_cancel_flag(correlation_id)

    except asyncio.CancelledError:
        await stream_service.update_stream_status(
            correlation_id,
            StreamStatus.CANCELLED,
        )
        await stream_service.clear_cancel_flag(correlation_id)
        await handle_cancellation(event_generator)
        raise
    except Exception as exc:

        def create_error_event_data(exception: Exception) -> str:
            error_evt = event_handler.error_event(correlation_id, exception)
            return event_handler.serialize(error_evt)

        event_data = await asyncio.to_thread(create_error_event_data, exc)
        await stream_service.push_event(
            correlation_id=correlation_id,
            event_data=event_data,
        )
        await stream_service.update_stream_status(
            correlation_id,
            StreamStatus.ERROR,
            error_message=str(exc),
        )
        await stream_service.clear_cancel_flag(correlation_id)
        raise


async def handle_cancellation(
    event_generator: AsyncGenerator[BaseModel, None],
) -> None:
    try:
        if hasattr(event_generator, "cancel"):
            await event_generator.cancel()
        elif hasattr(event_generator, "aclose"):
            await event_generator.aclose()
    except Exception:
        pass


async def start_stream_processing(
    *,
    stream_service: StreamService,
    task_manager: TaskManager,
    correlation_id: str,
    stream_type: str,
    event_generator: AsyncGenerator[BaseModel, None],
    event_handler: EventHandler,
    metadata: dict[str, Any] | None = None,
) -> asyncio.Task[Any]:
    return await task_manager.create_task(
        correlation_id=correlation_id,
        coro=process_stream(
            stream_service=stream_service,
            task_manager=task_manager,
            correlation_id=correlation_id,
            stream_type=stream_type,
            event_generator=event_generator,
            event_handler=event_handler,
            metadata=metadata,
        ),
        name=f"stream_processor_{correlation_id}",
    )
