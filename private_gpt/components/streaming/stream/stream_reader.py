import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Final

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.stream.event_handler import EventHandler
from private_gpt.components.streaming.stream_component import StreamComponent
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.streaming.providers.models import StreamMetadata

logger = logging.getLogger(__name__)


DEFAULT_BLOCK_MS: Final[int] = 1000
BATCH_SIZE: Final[int] = 5
TERMINAL_STATUSES: Final[set[StreamStatus]] = {
    StreamStatus.COMPLETED,
    StreamStatus.CANCELLED,
    StreamStatus.ERROR,
}


class StreamConsumer:
    """Consumer that receives events for a specific stream."""

    def __init__(
        self,
        correlation_id: str,
        event_handler: EventHandler,
    ):
        self.correlation_id = correlation_id
        self.event_handler = event_handler
        self.queue: asyncio.Queue[BaseModel | None] = asyncio.Queue()
        self.ref_count = 1

    async def send(self, event: BaseModel) -> None:
        self.queue.put_nowait(event)

    async def close(self) -> None:
        await self.queue.put(None)


class StreamState:
    """Tracks state for a multiplexed stream."""

    def __init__(self, last_id: str = "0"):
        self.last_id = last_id
        self.last_flush_time = asyncio.get_event_loop().time()
        self.cached_events: list[BaseModel] | None = None


@singleton
class StreamReader:
    """Reads events from Redis and deserializes them."""

    @inject
    def __init__(self, stream_component: StreamComponent):
        self.stream_service = stream_component.stream

    async def read_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        """Read events from Redis and deserialize them."""
        raw_events, next_last_id = await self.stream_service.read_events(
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )

        def sync_deserialization() -> list[BaseModel]:
            events = []
            for raw_data in raw_events:
                try:
                    event = event_handler.deserialize(raw_data)
                    events.append(event)
                except Exception as e:
                    logger.error(f"Error deserializing event: {e}")
                    continue
            return events

        final_events = sync_deserialization()
        return final_events, next_last_id

    async def check_terminal_status(
        self,
        correlation_id: str,
        event_handler: EventHandler,
        cached_events: list[BaseModel] | None,
        last_flush_time: float,
        cache_flush_interval: int,
    ) -> bool:
        """Check if stream has reached terminal status."""
        current_time = asyncio.get_event_loop().time()

        if cached_events and (current_time - last_flush_time) >= cache_flush_interval:
            try:
                status = await event_handler.get_current_status(cached_events[-1])
                if status in TERMINAL_STATUSES:
                    return True
            except Exception as e:
                logger.error(f"Error checking cached status for {correlation_id}: {e}")

        try:
            metadata: StreamMetadata | None = (
                await self.stream_service.get_stream_metadata(correlation_id)
            )
            if metadata and metadata.status in TERMINAL_STATUSES:
                return True
        except Exception as e:
            logger.error(f"Error checking metadata for {correlation_id}: {e}")

        return False

    async def stream_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        block_ms: int | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[BaseModel, None]:
        """Stream events as an async generator."""
        stop_event = stop_event or asyncio.Event()
        block_ms = block_ms if block_ms is not None else DEFAULT_BLOCK_MS
        cache_flush_interval: Final[int] = 5 * block_ms
        event_queue: asyncio.Queue[list[BaseModel] | None] = asyncio.Queue()

        async def produce_events(current_last_id: str) -> None:
            try:
                cached_events: list[Any] | None = None
                last_flush = asyncio.get_event_loop().time()

                while not stop_event.is_set():
                    events, current_last_id = await self.read_events(
                        event_handler=event_handler,
                        correlation_id=correlation_id,
                        last_id=current_last_id,
                        block_ms=block_ms,
                    )

                    if events:
                        cached_events = events
                        last_flush = asyncio.get_event_loop().time()
                        await event_queue.put(events)
                    elif await self.check_terminal_status(
                        correlation_id,
                        event_handler,
                        cached_events,
                        last_flush,
                        cache_flush_interval,
                    ):
                        break

                events, _ = await self.read_events(
                    event_handler=event_handler,
                    correlation_id=correlation_id,
                    last_id=current_last_id,
                )

                if events:
                    await event_queue.put(events)

            except Exception as e:
                logger.error(f"Producer error for {correlation_id}: {e}", exc_info=True)
            finally:
                await event_queue.put(None)

        async def get() -> AsyncGenerator[BaseModel, None]:
            producer_task = asyncio.create_task(produce_events(last_id))

            try:
                while not stop_event.is_set():
                    events = await event_queue.get()
                    if events is None:
                        break
                    for event in events:
                        yield event

            finally:
                producer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer_task

                while not event_queue.empty():
                    events = event_queue.get_nowait()
                    if events is None:
                        break
                    for event in events:
                        yield event

        return get()


class StreamMultiplexer:
    """Multiplexes multiple stream consumers onto a single worker."""

    def __init__(self, stream_reader: StreamReader):
        self.stream_reader = stream_reader
        self.consumers: dict[str, list[StreamConsumer]] = defaultdict(list)
        self.states: dict[str, StreamState] = {}
        self.worker_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._cache_flush_interval = 5 * DEFAULT_BLOCK_MS

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.shutdown_event.clear()
            self.worker_task = asyncio.create_task(self._run_worker())
            logger.info("Stream multiplexer started")

    async def stop(self) -> None:
        self.shutdown_event.set()
        if self.worker_task:
            await self.worker_task
        logger.info("Stream multiplexer stopped")

    async def add_consumer(
        self,
        correlation_id: str,
        event_handler: EventHandler,
    ) -> StreamConsumer:
        async with self._lock:
            existing_consumers = self.consumers.get(correlation_id, [])

            for consumer in existing_consumers:
                if isinstance(consumer.event_handler, type(event_handler)):
                    consumer.ref_count += 1
                    return consumer

            consumer = StreamConsumer(correlation_id, event_handler)
            self.consumers[correlation_id].append(consumer)

            if correlation_id not in self.states:
                self.states[correlation_id] = StreamState()

            return consumer

    async def remove_consumer(self, consumer: StreamConsumer) -> None:
        async with self._lock:
            correlation_id = consumer.correlation_id
            consumer.ref_count -= 1

            if consumer.ref_count <= 0:
                if correlation_id in self.consumers:
                    self.consumers[correlation_id].remove(consumer)

                    if not self.consumers[correlation_id]:
                        del self.consumers[correlation_id]
                        if correlation_id in self.states:
                            del self.states[correlation_id]

                await consumer.close()

    async def _run_worker(self) -> None:
        try:
            while not self.shutdown_event.is_set():
                async with self._lock:
                    correlation_ids = list(self.states.keys())

                if not correlation_ids:
                    await asyncio.sleep(1)
                    continue

                tasks = [
                    self._process_stream(correlation_id)
                    for correlation_id in correlation_ids
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
        finally:
            async with self._lock:
                for consumer_list in self.consumers.values():
                    for consumer in consumer_list:
                        await consumer.close()

    async def _process_stream(self, correlation_id: str) -> None:
        try:
            async with self._lock:
                if correlation_id not in self.states:
                    return
                state = self.states[correlation_id]
                consumer_list = self.consumers.get(correlation_id, [])
                if not consumer_list:
                    return

            events, new_last_id = await self.stream_reader.read_events(
                event_handler=consumer_list[0].event_handler,
                correlation_id=correlation_id,
                last_id=state.last_id,
            )

            current_time = asyncio.get_event_loop().time()

            if events:
                async with self._lock:
                    state.last_id = new_last_id
                    state.last_flush_time = current_time
                    state.cached_events = events

                send_tasks = []
                for consumer in consumer_list:
                    for event in events:
                        send_tasks.append(consumer.send(event))

                await asyncio.gather(*send_tasks)

            elif await self.stream_reader.check_terminal_status(
                correlation_id,
                consumer_list[0].event_handler,
                state.cached_events,
                state.last_flush_time,
                self._cache_flush_interval,
            ):
                await self._close_all_consumers(correlation_id)
                logger.info(f"Stream {correlation_id} reached terminal status")
            else:
                async with self._lock:
                    state.last_flush_time = current_time

        except Exception as e:
            logger.error(f"Error processing {correlation_id}: {e}", exc_info=True)

    async def _close_all_consumers(self, correlation_id: str) -> None:
        async with self._lock:
            consumer_list = self.consumers.get(correlation_id, [])
            for consumer in consumer_list:
                await consumer.close()

            if correlation_id in self.consumers:
                del self.consumers[correlation_id]
            if correlation_id in self.states:
                del self.states[correlation_id]


@singleton
class AdaptiveStreamReader:
    """Adaptively switches between direct and multiplexed streaming based on load."""

    @inject
    def __init__(
        self,
        settings: Settings,
        stream_reader: StreamReader,
    ):
        self.stream_reader = stream_reader
        self.multiplexer: StreamMultiplexer | None = None
        self._active_count = 0
        self._lock = asyncio.Lock()
        self.enable_multiplexing = True
        self.multiplexing_threshold = settings.chat.multiplexing_threshold or None

    async def read_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        return await self.stream_reader.read_events(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )

    async def stream_events(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str = "0",
        block_ms: int | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[BaseModel, None]:
        async def gen() -> AsyncGenerator[BaseModel, None]:
            async with self._lock:
                self._active_count += 1
                should_multiplex = (
                    self.enable_multiplexing
                    and self.multiplexing_threshold is not None
                    and self._active_count >= self.multiplexing_threshold
                )

            try:
                if should_multiplex:
                    async for event in await self._stream_multiplexed(
                        event_handler, correlation_id, last_id, stop_event
                    ):
                        yield event
                else:
                    async for event in await self.stream_reader.stream_events(
                        event_handler, correlation_id, last_id, block_ms, stop_event
                    ):
                        yield event
            finally:
                async with self._lock:
                    self._active_count -= 1

        return gen()

    async def _stream_multiplexed(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str,
        stop_event: asyncio.Event | None,
    ) -> AsyncGenerator[BaseModel, None]:
        async def gen() -> AsyncGenerator[BaseModel, None]:
            if self.multiplexer is None:
                self.multiplexer = StreamMultiplexer(self.stream_reader)
                await self.multiplexer.start()
                logger.info("Multiplexed streaming activated")

            consumer = await self.multiplexer.add_consumer(
                correlation_id, event_handler
            )

            try:
                if last_id != "0":
                    initial_events, _ = await self.read_events(
                        event_handler, correlation_id, "0", None, 0
                    )
                    for initial_event in initial_events:
                        yield initial_event

                events_processed = 0
                while True:
                    if stop_event and stop_event.is_set():
                        break

                    try:
                        event: Any | None = await asyncio.wait_for(
                            consumer.queue.get(), timeout=1.0
                        )
                        if event is None:
                            break
                        yield event

                        events_processed += 1
                        if events_processed % BATCH_SIZE == 0:
                            await asyncio.sleep(0)

                    except TimeoutError:
                        continue

            finally:
                await self.multiplexer.remove_consumer(consumer)

        return gen()

    async def shutdown(self) -> None:
        if self.multiplexer:
            await self.multiplexer.stop()
            self.multiplexer = None
