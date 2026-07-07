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
    """A single consumer receiving events for one correlation id.

    Events are buffered in an unbounded ``asyncio.Queue``.  A ``None`` entry
    is the sentinel that signals the consumer the stream is finished.
    """

    def __init__(self, correlation_id: str, event_handler: EventHandler):
        self.correlation_id = correlation_id
        self.event_handler = event_handler
        self.queue: asyncio.Queue[BaseModel | None] = asyncio.Queue()
        self.ref_count = 1

    def close(self) -> None:
        """Signal the consumer that no more events will arrive."""
        self.queue.put_nowait(None)


class StreamState:
    """Read position and cache for a single multiplexed stream."""

    def __init__(self, last_id: str = "0"):
        self.last_id = last_id
        self.last_flush_time = asyncio.get_event_loop().time()
        self.cached_events: list[BaseModel] | None = None


@singleton
class StreamReader:
    """Reads events from the backend (Redis or in-memory) and deserializes them.

    On Redis, ``read_events`` with ``block_ms > 0`` issues ``XREAD BLOCK`` which
    parks the connection — the event loop stays free.  On the in-memory backend
    there is no real blocking primitive, so ``block_ms > 0`` is approximated
    with ``asyncio.sleep`` to yield control.
    ``block_ms=None`` is always non-blocking (used for catch-up / drain reads).
    """

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
        raw_events, next_last_id = await self.stream_service.read_events(
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )

        events: list[BaseModel] = []
        for raw_data in raw_events:
            try:
                events.append(event_handler.deserialize(raw_data))
            except Exception as e:
                logger.error(f"Error deserializing event: {e}")
        return events, next_last_id

    async def check_terminal_status(
        self,
        correlation_id: str,
        event_handler: EventHandler,
        cached_events: list[BaseModel] | None,
        last_flush_time: float,
        cache_flush_interval: int,
    ) -> bool:
        """Return True when the stream has reached a terminal status."""
        current_time = asyncio.get_event_loop().time()

        if cached_events and (current_time - last_flush_time) >= cache_flush_interval:
            try:
                status = await event_handler.get_current_status(cached_events[-1])
                if status in TERMINAL_STATUSES:
                    return True
            except Exception as e:
                logger.error(f"Error checking cached status for {correlation_id}: {e}")

        try:
            metadata: (
                StreamMetadata | None
            ) = await self.stream_service.get_stream_metadata(correlation_id)
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
        """Stream events as an async generator (direct, non-multiplexed path).

        A producer task reads from the backend in blocking mode and feeds an
        internal queue; the generator drains the queue and yields individual
        events.  After the terminal status is detected a final non-blocking
        drain read is performed so no event published between the last read
        and the terminal signal is lost.
        """
        stop_event = stop_event or asyncio.Event()
        block_ms = block_ms if block_ms is not None else DEFAULT_BLOCK_MS
        cache_flush_interval: Final[int] = 5 * block_ms
        event_queue: asyncio.Queue[list[BaseModel] | None] = asyncio.Queue()

        async def produce_events(current_last_id: str) -> None:
            try:
                cached_events: list[BaseModel] | None = None
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
    """Multiplexes many stream consumers onto a single worker task.

    Each correlation id has one ``StreamState`` (read position) shared by all
    consumers on that stream.  The worker loops over every active stream,
    issues a blocking ``read_events`` and dispatches the batch to every
    current consumer.

    Key invariants
    ---------------
    * Dispatch and ``last_id`` advance happen atomically under the lock, and
      the consumer list is re-fetched immediately before dispatch.  This means
      a consumer that joins while the worker is blocked inside ``read_events``
      still receives the batch about to be dispatched.
    * ``add_consumer`` accepts a ``last_id`` and initialises the stream state
      from it, so a reconnecting consumer never receives a replay from ``"0"``.
    * ``stop`` cancels the worker, clears all state and closes every consumer;
      a subsequent ``start`` begins from a clean slate.
    """

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
        """Cancel the worker, clear all state, and close every consumer."""
        self.shutdown_event.set()
        task = self.worker_task
        self.worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        async with self._lock:
            all_consumers = [c for cl in self.consumers.values() for c in cl]
            self.consumers.clear()
            self.states.clear()
        for consumer in all_consumers:
            consumer.close()

    async def add_consumer(
        self,
        correlation_id: str,
        event_handler: EventHandler,
        last_id: str = "0",
    ) -> StreamConsumer:
        """Register a consumer for *correlation_id*.

        If a consumer with the same handler type already exists its ref-count
        is incremented and the existing consumer is returned (reconnect case).
        Otherwise a new consumer is created.  When this is the first consumer
        for the stream, the read position is initialised to *last_id*.
        """
        async with self._lock:
            for consumer in self.consumers.get(correlation_id, []):
                if type(consumer.event_handler) is type(event_handler):
                    consumer.ref_count += 1
                    return consumer

            consumer = StreamConsumer(correlation_id, event_handler)
            self.consumers[correlation_id].append(consumer)

            if correlation_id not in self.states:
                self.states[correlation_id] = StreamState(last_id=last_id)

            return consumer

    async def remove_consumer(self, consumer: StreamConsumer) -> None:
        should_close = False
        async with self._lock:
            correlation_id = consumer.correlation_id
            consumer.ref_count -= 1

            if consumer.ref_count <= 0:
                cl = self.consumers.get(correlation_id)
                if cl and consumer in cl:
                    cl.remove(consumer)
                if (
                    correlation_id in self.consumers
                    and not self.consumers[correlation_id]
                ):
                    del self.consumers[correlation_id]
                    self.states.pop(correlation_id, None)
                should_close = True

        if should_close:
            consumer.close()

    async def _run_worker(self) -> None:
        try:
            while not self.shutdown_event.is_set():
                async with self._lock:
                    correlation_ids = list(self.states.keys())

                if not correlation_ids:
                    await asyncio.sleep(0.5)
                    continue

                tasks = [self._process_stream(cid) for cid in correlation_ids]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
        finally:
            for cl in list(self.consumers.values()):
                for consumer in cl:
                    consumer.close()

    async def _process_stream(self, correlation_id: str) -> None:
        try:
            async with self._lock:
                if correlation_id not in self.states:
                    return
                state = self.states[correlation_id]
                consumer_list = list(self.consumers.get(correlation_id, []))
                if not consumer_list:
                    return

            events, new_last_id = await self.stream_reader.read_events(
                event_handler=consumer_list[0].event_handler,
                correlation_id=correlation_id,
                last_id=state.last_id,
                block_ms=DEFAULT_BLOCK_MS,
            )

            current_time = asyncio.get_event_loop().time()

            if events:
                await self._dispatch(
                    correlation_id, state, events, new_last_id, current_time
                )

            elif await self.stream_reader.check_terminal_status(
                correlation_id,
                consumer_list[0].event_handler,
                state.cached_events,
                state.last_flush_time,
                self._cache_flush_interval,
            ):
                drain_events, drain_last_id = await self.stream_reader.read_events(
                    event_handler=consumer_list[0].event_handler,
                    correlation_id=correlation_id,
                    last_id=state.last_id,
                    block_ms=None,
                )
                if drain_events:
                    await self._dispatch(
                        correlation_id, state, drain_events, drain_last_id, current_time
                    )
                await self._close_all_consumers(correlation_id)
                logger.info(f"Stream {correlation_id} reached terminal status")
            else:
                async with self._lock:
                    if correlation_id in self.states:
                        self.states[correlation_id].last_flush_time = current_time

        except Exception as e:
            logger.error(f"Error processing {correlation_id}: {e}", exc_info=True)

    async def _dispatch(
        self,
        correlation_id: str,
        state: StreamState,
        events: list[BaseModel],
        new_last_id: str,
        current_time: float,
    ) -> None:
        """Dispatch *events* to every current consumer and advance ``last_id``.

        The consumer list is re-fetched under the lock so that consumers that
        joined or left during the preceding blocking ``read_events`` are
        reflected.  Dispatch and ``last_id`` advance are atomic: no consumer
        can join between the two and miss the batch.
        """
        async with self._lock:
            if correlation_id not in self.states:
                return
            consumer_list = list(self.consumers.get(correlation_id, []))
            if not consumer_list:
                return
            for consumer in consumer_list:
                for event in events:
                    consumer.queue.put_nowait(event)
            state.last_id = new_last_id
            state.last_flush_time = current_time
            state.cached_events = events

    async def _close_all_consumers(self, correlation_id: str) -> None:
        async with self._lock:
            consumer_list = list(self.consumers.get(correlation_id, []))
            self.consumers.pop(correlation_id, None)
            self.states.pop(correlation_id, None)
        for consumer in consumer_list:
            consumer.close()


@singleton
class AdaptiveStreamReader:
    """Switches between the direct reader and the multiplexer based on load.

    When the number of concurrently active streams reaches
    ``multiplexing_threshold`` the multiplexed path is used; otherwise the
    direct ``StreamReader.stream_events`` path is used.  Both paths must
    produce identical results for the same stream and ``last_id``.
    """

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
                correlation_id, event_handler, last_id=last_id
            )

            try:
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
