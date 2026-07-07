import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Final

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
# How often (seconds) to poll for terminal status when no events arrive.
# Halves Redis metadata round-trips vs. checking every block cycle (1 s).
STATUS_CHECK_INTERVAL: Final[float] = 2.0
TERMINAL_STATUSES: Final[set[StreamStatus]] = {
    StreamStatus.COMPLETED,
    StreamStatus.CANCELLED,
    StreamStatus.ERROR,
}


class StreamBroadcast:
    """Append-only batch log that broadcasts to N concurrent readers.

    A single producer calls ``push`` / ``close`` (both synchronous); each
    subscriber holds an integer *cursor* into ``_batches`` and parks on a
    ``asyncio.Future`` until new data arrives.  One ``push`` resolves *all*
    pending futures in a single pass — no per-subscriber queues.

    Safety: all operations are synchronous except ``read_from``, so no lock is
    needed (asyncio cooperative threading guarantees no interleaving between
    non-await statements).
    """

    def __init__(self) -> None:
        self._batches: list[list[BaseModel] | None] = []
        self._waiters: list[asyncio.Future[None]] = []

    def _notify(self) -> None:
        waiters, self._waiters = self._waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    def push(self, batch: list[BaseModel]) -> None:
        self._batches.append(batch)
        self._notify()

    def close(self) -> None:
        """Append the terminal sentinel and wake all parked readers."""
        self._batches.append(None)
        self._notify()

    def next_cursor(self) -> int:
        """Cursor for a subscriber joining right now (no backfill)."""
        return len(self._batches)

    @property
    def is_closed(self) -> bool:
        return bool(self._batches) and self._batches[-1] is None

    async def read_from(
        self,
        cursor: int,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[BaseModel, None]:
        """Yield events starting at *cursor*, blocking until new ones arrive."""
        while True:
            # Drain already-available batches.  Safe without a lock: _batches is
            # append-only and list operations are atomic between await points.
            while cursor < len(self._batches):
                batch = self._batches[cursor]
                cursor += 1
                if batch is None:
                    return
                for event in batch:
                    yield event

            if stop_event and stop_event.is_set():
                return

            # Park until the producer pushes more data.
            fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._waiters.append(fut)
            try:
                await asyncio.wait_for(fut, timeout=1.0)
            except TimeoutError:
                pass  # Re-check stop_event and drain any new batches
            finally:
                # Clean up if push() hasn't already swapped us out.
                if fut in self._waiters:
                    self._waiters.remove(fut)


class StreamConsumer:
    """A subscriber on a ``StreamBroadcast`` with an individual read cursor."""

    def __init__(
        self,
        correlation_id: str,
        event_handler: EventHandler,
        broadcast: StreamBroadcast,
        cursor: int,
    ) -> None:
        self.correlation_id = correlation_id
        self.event_handler = event_handler
        self.broadcast = broadcast
        self.cursor = cursor  # Index into broadcast._batches


class StreamState:
    """Read position and broadcast log for a single multiplexed stream."""

    def __init__(self, last_id: str = "0") -> None:
        self.last_id = last_id
        self.cached_events: list[BaseModel] | None = None
        # 0.0 → check on the first idle cycle (ancient enough to pass the gate)
        self.last_status_check: float = 0.0
        self.broadcast = StreamBroadcast()


@singleton
class StreamReader:
    """Reads events from the backend (Redis or in-memory) and deserializes them.

    On Redis, ``read_events`` with ``block_ms > 0`` issues ``XREAD BLOCK`` which
    parks the connection — the event loop stays free.  On the in-memory backend
    the provider uses an ``asyncio.Event`` to avoid busy-waiting.
    ``block_ms=None`` is always non-blocking (used for catch-up / drain reads).
    """

    @inject
    def __init__(self, stream_component: StreamComponent) -> None:
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
    ) -> bool:
        """Return True when the stream has reached a terminal status.

        Tries the last cached event first; only hits Redis when that check is
        inconclusive, avoiding a round-trip when the cached event already
        carries terminal state.
        """
        if cached_events:
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
        """Stream events as an async generator (direct, non-multiplexed path).

        A producer task reads from the backend in blocking mode and feeds an
        internal queue; the generator drains the queue and yields individual
        events.  After the terminal status is detected a final non-blocking
        drain read is performed so no event published between the last read
        and the terminal signal is lost.
        """
        stop_event = stop_event or asyncio.Event()
        block_ms = block_ms if block_ms is not None else DEFAULT_BLOCK_MS
        event_queue: asyncio.Queue[list[BaseModel] | None] = asyncio.Queue()

        async def produce_events(current_last_id: str) -> None:
            try:
                cached_events: list[BaseModel] | None = None
                last_status_check: float = 0.0  # 0 → check on first idle cycle

                while not stop_event.is_set():
                    events, current_last_id = await self.read_events(
                        event_handler=event_handler,
                        correlation_id=correlation_id,
                        last_id=current_last_id,
                        block_ms=block_ms,
                    )

                    if events:
                        cached_events = events
                        await event_queue.put(events)
                    else:
                        current_time = asyncio.get_event_loop().time()
                        if (current_time - last_status_check) >= STATUS_CHECK_INTERVAL:
                            if await self.check_terminal_status(
                                correlation_id,
                                event_handler,
                                cached_events,
                            ):
                                break
                            last_status_check = current_time

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

    Each correlation id has one ``StreamState`` containing a ``StreamBroadcast``
    shared by all subscribers.  The worker calls ``_process_stream`` once per
    active stream per cycle: one Redis read, one ``broadcast.push`` — no
    per-consumer queues.

    Key invariants
    ---------------
    * ``broadcast.push`` is synchronous and resolves all parked futures in one
      pass, so all N subscribers wake atomically after a single call.
    * State is updated under the lock; ``broadcast.push`` happens outside it so
      the lock is held for as short a time as possible.
    * ``close`` on the broadcast terminates all ``read_from`` generators, making
      ``stop`` and terminal-status cleanup self-contained.
    """

    def __init__(self, stream_reader: StreamReader) -> None:
        self.stream_reader = stream_reader
        self.consumers: dict[str, list[StreamConsumer]] = defaultdict(list)
        self.states: dict[str, StreamState] = {}
        self.worker_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.worker_task is None or self.worker_task.done():
            self.shutdown_event.clear()
            self.worker_task = asyncio.create_task(self._run_worker())
            logger.info("Stream multiplexer started")

    async def stop(self) -> None:
        """Cancel the worker, clear all state, and close every broadcast."""
        self.shutdown_event.set()
        task = self.worker_task
        self.worker_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        async with self._lock:
            states_to_close = list(self.states.values())
            self.consumers.clear()
            self.states.clear()

        for state in states_to_close:
            state.broadcast.close()

    async def add_consumer(
        self,
        correlation_id: str,
        event_handler: EventHandler,
        last_id: str = "0",
    ) -> StreamConsumer:
        """Register a new subscriber for *correlation_id*.

        Each call returns a distinct ``StreamConsumer`` with its own cursor so
        all concurrent subscribers receive every event independently (true
        fan-out).  When this is the first consumer for the stream the read
        position is initialised to *last_id*.
        """
        async with self._lock:
            if correlation_id not in self.states:
                self.states[correlation_id] = StreamState(last_id=last_id)

            state = self.states[correlation_id]
            # last_id="0" means "from the start" — replay every batch already in
            # the broadcast so a late joiner never misses history (e.g. the
            # content_block_start that arrived before it connected).
            # A specific last_id means the consumer is already caught up to that
            # position, so start at the current end and receive only new batches.
            cursor = 0 if last_id == "0" else state.broadcast.next_cursor()
            consumer = StreamConsumer(
                correlation_id, event_handler, state.broadcast, cursor
            )
            self.consumers[correlation_id].append(consumer)
            return consumer

    async def remove_consumer(self, consumer: StreamConsumer) -> None:
        broadcast_to_close: StreamBroadcast | None = None
        async with self._lock:
            correlation_id = consumer.correlation_id
            cl = self.consumers.get(correlation_id)
            if cl and consumer in cl:
                cl.remove(consumer)
            if correlation_id in self.consumers and not self.consumers[correlation_id]:
                del self.consumers[correlation_id]
                state = self.states.pop(correlation_id, None)
                if state:
                    broadcast_to_close = state.broadcast

        if broadcast_to_close is not None:
            broadcast_to_close.close()

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
            async with self._lock:
                states_to_close = list(self.states.values())
                self.consumers.clear()
                self.states.clear()
            for state in states_to_close:
                state.broadcast.close()

    async def _process_stream(self, correlation_id: str) -> None:
        try:
            async with self._lock:
                if correlation_id not in self.states:
                    return
                state = self.states[correlation_id]
                consumer_list = list(self.consumers.get(correlation_id, []))
                if not consumer_list:
                    return
                event_handler = consumer_list[0].event_handler

            events, new_last_id = await self.stream_reader.read_events(
                event_handler=event_handler,
                correlation_id=correlation_id,
                last_id=state.last_id,
                block_ms=DEFAULT_BLOCK_MS,
            )

            current_time = asyncio.get_event_loop().time()

            if events:
                await self._dispatch(
                    correlation_id, state, events, new_last_id, current_time
                )

            elif (current_time - state.last_status_check) >= STATUS_CHECK_INTERVAL:
                state.last_status_check = current_time
                if await self.stream_reader.check_terminal_status(
                    correlation_id,
                    event_handler,
                    state.cached_events,
                ):
                    drain_events, drain_last_id = await self.stream_reader.read_events(
                        event_handler=event_handler,
                        correlation_id=correlation_id,
                        last_id=state.last_id,
                        block_ms=None,
                    )
                    if drain_events:
                        await self._dispatch(
                            correlation_id,
                            state,
                            drain_events,
                            drain_last_id,
                            current_time,
                        )
                    await self._close_all_consumers(correlation_id)
                    logger.info(f"Stream {correlation_id} reached terminal status")

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
        """Advance stream state then broadcast *events* to all subscribers.

        State is updated under the lock; ``broadcast.push`` is synchronous and
        happens outside so the lock is held for the minimum time.
        """
        async with self._lock:
            if correlation_id not in self.states:
                return
            if not self.consumers.get(correlation_id):
                return
            state.last_id = new_last_id
            state.cached_events = events
            state.last_status_check = current_time

        state.broadcast.push(events)

    async def _close_all_consumers(self, correlation_id: str) -> None:
        async with self._lock:
            state = self.states.pop(correlation_id, None)
            self.consumers.pop(correlation_id, None)
        if state:
            state.broadcast.close()


@singleton
class AdaptiveStreamReader:
    """Routes stream consumers between direct and multiplexed paths.

    Direct path: one XREAD BLOCK per consumer → one Redis connection held.
    Multiplexed path: one shared worker for all active streams on this process.

    When concurrent direct-path readers reach ``multiplexing_threshold`` new
    consumers are routed to the multiplexer instead of opening another Redis
    connection.  When the multiplexer drains to zero active streams it is
    stopped, so load-shedding is fully reversible — consumers that arrive when
    load is back below the threshold go direct again.

    ``multiplexing_threshold=None`` disables multiplexing entirely.
    ``enable_multiplexing=False`` is a hard kill-switch (e.g. for tests).
    """

    @inject
    def __init__(
        self,
        settings: Settings,
        stream_reader: StreamReader,
    ) -> None:
        self.stream_reader = stream_reader
        self.multiplexer: StreamMultiplexer | None = None
        self._lock = asyncio.Lock()
        self.enable_multiplexing = True
        self.multiplexing_threshold: int | None = settings.chat.multiplexing_threshold
        # Number of consumers currently on the direct path.
        # Each one holds an XREAD BLOCK connection; this is what we throttle.
        self._direct_count = 0

    def _should_multiplex(self) -> bool:
        return (
            self.enable_multiplexing
            and self.multiplexing_threshold is not None
            and self._direct_count >= self.multiplexing_threshold
        )

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
            # Decide path under the lock so _direct_count is consistent.
            async with self._lock:
                use_mux = self._should_multiplex()
                if not use_mux:
                    self._direct_count += 1

            try:
                if use_mux:
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
                if not use_mux:
                    async with self._lock:
                        self._direct_count -= 1

        return gen()

    async def _stream_multiplexed(
        self,
        event_handler: EventHandler,
        correlation_id: str,
        last_id: str,
        stop_event: asyncio.Event | None,
    ) -> AsyncGenerator[BaseModel, None]:
        async def gen() -> AsyncGenerator[BaseModel, None]:
            # Start the multiplexer on first use; capture the reference so the
            # finally block targets the right instance even if it is replaced.
            async with self._lock:
                if self.multiplexer is None:
                    self.multiplexer = StreamMultiplexer(self.stream_reader)
                    await self.multiplexer.start()
                    logger.info("Multiplexer started (threshold reached)")
                mux = self.multiplexer

            consumer = await mux.add_consumer(
                correlation_id, event_handler, last_id=last_id
            )
            try:
                async for event in consumer.broadcast.read_from(
                    cursor=consumer.cursor,
                    stop_event=stop_event,
                ):
                    yield event
            finally:
                await mux.remove_consumer(consumer)
                # Stop the multiplexer when it has no more active streams so the
                # next consumer below the threshold can go direct again.
                async with self._lock:
                    if self.multiplexer is mux and not mux.states:
                        await mux.stop()
                        self.multiplexer = None
                        logger.info("Multiplexer stopped (no active streams)")

        return gen()

    async def shutdown(self) -> None:
        async with self._lock:
            mux, self.multiplexer = self.multiplexer, None
        if mux:
            await mux.stop()
