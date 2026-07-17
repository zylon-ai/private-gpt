"""Tests for StreamMultiplexer and the streaming readers.

The core strategy is **differential testing**: each scenario is run through the
direct path (StreamReader.stream_events) AND the multiplexed path
(StreamMultiplexer), and the two results must be identical. If they ever
diverge, the multiplexer is broken.

Section layout
--------------
1. Differential tests  — direct == multiplexed for every scenario.
2. Multiplexer concurrency — late joiners, concurrent remove, multi-consumer.
3. Terminal detection — events in the terminal gap are drained.
4. Lifecycle — stop/restart leave no stale state.
5. Performance — worker does not busy-poll.
6. Backend block semantics — block_ms=None is non-blocking.
"""

import asyncio
import json
import random
from typing import Any

import pytest
from pydantic import BaseModel

from private_gpt.components.streaming.providers.in_memory_stream_service import (
    InMemoryStreamService,
)
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.providers.stream_service import StreamService
from private_gpt.components.streaming.stream.stream_reader import (
    DEFAULT_BLOCK_MS,
    StreamConsumer,
    StreamMultiplexer,
    StreamReader,
)

STRESS_N = 500


# ──────────────────────────── helpers / models ────────────────────────────────


class SimpleEvent(BaseModel):
    n: int


class SimpleEventHandler:
    def serialize(self, event: BaseModel) -> str:
        return event.model_dump_json()

    def deserialize(self, data: str) -> BaseModel:
        return SimpleEvent.model_validate_json(data)

    async def get_current_status(self, event: BaseModel) -> StreamStatus | None:
        return None

    def error_event(self, correlation_id: str, error: Exception) -> BaseModel:
        return SimpleEvent(n=-1)


class SimpleEventHandler2(SimpleEventHandler):
    """Distinct type so add_consumer creates a separate consumer."""


class MockStreamComponent:
    def __init__(self, service: StreamService) -> None:
        self.stream = service


async def collect_from_consumer(
    consumer: StreamConsumer,
    timeout: float = 10.0,
) -> list[BaseModel]:
    """Collect all events from a consumer's broadcast until it is closed."""
    events: list[BaseModel] = []

    async def drain() -> None:
        async for event in consumer.broadcast.read_from(cursor=consumer.cursor):
            events.append(event)

    await asyncio.wait_for(drain(), timeout=timeout)
    return events


async def run_direct(
    stream_reader: StreamReader,
    handler: SimpleEventHandler,
    cid: str,
    last_id: str = "0",
) -> list[BaseModel]:
    received: list[BaseModel] = []
    gen = await stream_reader.stream_events(
        event_handler=handler,
        correlation_id=cid,
        last_id=last_id,
    )
    async for event in gen:
        received.append(event)
    return received


async def run_multiplexed(
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    cid: str,
    last_id: str = "0",
) -> list[BaseModel]:
    consumer = await multiplexer.add_consumer(cid, handler, last_id=last_id)
    return await collect_from_consumer(consumer, timeout=30.0)


async def push_n_and_complete(
    service: StreamService,
    cid: str,
    n: int,
    *,
    burst: int = 10,
    max_jitter: float = 0.0,
) -> None:
    for i in range(n):
        await service.push_event(cid, json.dumps({"n": i}))
        if (i + 1) % burst == 0:
            await asyncio.sleep(random.uniform(0, max_jitter))
    await asyncio.sleep(random.uniform(0, max_jitter * 2))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)


def assert_ordered_0_to_n(events: list[BaseModel], n: int) -> None:
    ns = [e.n for e in events]  # type: ignore[attr-defined]
    missing = sorted(set(range(n)) - set(ns))
    duplicates = sorted({x for x in ns if ns.count(x) > 1})
    assert not missing, f"missing {len(missing)} events: {missing[:10]}"
    assert not duplicates, f"duplicates: {duplicates[:10]}"
    assert ns == sorted(ns), "events arrived out of order"
    assert len(ns) == n


# ──────────────────────────── fixtures ───────────────────────────────────────


@pytest.fixture
def service() -> StreamService:
    return InMemoryStreamService()


@pytest.fixture
def stream_reader(service: StreamService) -> StreamReader:
    return StreamReader(MockStreamComponent(service))  # type: ignore[arg-type]


@pytest.fixture
def multiplexer(stream_reader: StreamReader) -> StreamMultiplexer:
    return StreamMultiplexer(stream_reader)


@pytest.fixture
def handler() -> SimpleEventHandler:
    return SimpleEventHandler()


@pytest.fixture
def handler2() -> SimpleEventHandler2:
    return SimpleEventHandler2()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Differential tests — direct path must equal multiplexed path
# ═══════════════════════════════════════════════════════════════════════════════


async def test_both_paths_prepopulated(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.push_event(cid, json.dumps({"n": 2}))
    await service.push_event(cid, json.dumps({"n": 3}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    try:
        mux = await run_multiplexed(multiplexer, handler, cid)
    finally:
        await multiplexer.stop()
    direct = await run_direct(stream_reader, handler, cid)

    assert [e.n for e in mux] == [e.n for e in direct] == [1, 2, 3]  # type: ignore[attr-defined]


async def test_both_paths_empty_stream(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    cid = await service.create_stream("test")
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    try:
        mux = await run_multiplexed(multiplexer, handler, cid)
    finally:
        await multiplexer.stop()
    direct = await run_direct(stream_reader, handler, cid)

    assert mux == direct == []


async def test_both_paths_reconnect_from_mid(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    """A consumer that already has event n=1 (it knows last_id=e1_id) must
    receive only events after that point — never a replay from "0"."""
    cid = await service.create_stream("test")
    e1_id = await service.push_event(cid, json.dumps({"n": 1}))
    await service.push_event(cid, json.dumps({"n": 2}))
    await service.push_event(cid, json.dumps({"n": 3}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    try:
        mux = await run_multiplexed(multiplexer, handler, cid, last_id=e1_id)
    finally:
        await multiplexer.stop()
    direct = await run_direct(stream_reader, handler, cid, last_id=e1_id)

    assert [e.n for e in mux] == [e.n for e in direct] == [2, 3]  # type: ignore[attr-defined]


async def test_both_paths_reconnect_from_tail(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    """Consumer is caught up to the last event; both paths return nothing."""
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.push_event(cid, json.dumps({"n": 2}))
    e3_id = await service.push_event(cid, json.dumps({"n": 3}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    try:
        mux = await run_multiplexed(multiplexer, handler, cid, last_id=e3_id)
    finally:
        await multiplexer.stop()
    direct = await run_direct(stream_reader, handler, cid, last_id=e3_id)

    assert mux == direct == []


async def test_both_paths_live_producer(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    """Events arrive while both consumers are already blocking on the stream.
    Both paths must receive every event in order."""
    cid = await service.create_stream("test")

    await multiplexer.start()
    try:
        mux_task = asyncio.create_task(run_multiplexed(multiplexer, handler, cid))
        direct_task = asyncio.create_task(run_direct(stream_reader, handler, cid))
        await asyncio.sleep(0.15)
        await push_n_and_complete(service, cid, 100, burst=10, max_jitter=0.003)
        mux = await asyncio.wait_for(mux_task, timeout=30.0)
        direct = await asyncio.wait_for(direct_task, timeout=30.0)
    finally:
        await multiplexer.stop()

    mux_ns = [e.n for e in mux]  # type: ignore[attr-defined]
    direct_ns = [e.n for e in direct]  # type: ignore[attr-defined]
    assert mux_ns == direct_ns
    assert_ordered_0_to_n(mux, 100)
    assert_ordered_0_to_n(direct, 100)


@pytest.mark.parametrize(
    ("burst", "max_jitter"),
    [
        (1, 0.0),
        (10, 0.003),
        (STRESS_N, 0.0),
    ],
    ids=["one-at-a-time", "small-bursts", "single-burst"],
)
async def test_both_paths_stress(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    burst: int,
    max_jitter: float,
) -> None:
    cid = await service.create_stream("test")
    await push_n_and_complete(
        service, cid, STRESS_N, burst=burst, max_jitter=max_jitter
    )

    await multiplexer.start()
    try:
        mux = await run_multiplexed(multiplexer, handler, cid)
    finally:
        await multiplexer.stop()
    direct = await run_direct(stream_reader, handler, cid)

    assert_ordered_0_to_n(mux, STRESS_N)
    assert_ordered_0_to_n(direct, STRESS_N)
    assert [e.n for e in mux] == [e.n for e in direct]  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Multiplexer concurrency
# ═══════════════════════════════════════════════════════════════════════════════


async def test_late_joiner_receives_inflight_batch(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    handler2: SimpleEventHandler2,
) -> None:
    """A consumer that joins while the worker is blocked inside read_events()
    must still receive the batch that the worker is about to dispatch."""
    cid = await service.create_stream("test")

    original_read = stream_reader.read_events
    late_box: dict[str, Any] = {}

    async def read_then_join(
        event_handler: Any,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        result = await original_read(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )
        if result[0] and "c2" not in late_box:
            late_box["c2"] = await multiplexer.add_consumer(cid, handler2)
        return result

    stream_reader.read_events = read_then_join  # type: ignore[method-assign]

    await multiplexer.start()
    c1 = await multiplexer.add_consumer(cid, handler)
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    try:
        recv1 = await collect_from_consumer(c1, timeout=10.0)
        c2 = late_box["c2"]
        recv2 = await collect_from_consumer(c2, timeout=10.0)
    finally:
        await multiplexer.stop()
        stream_reader.read_events = original_read  # type: ignore[method-assign]

    assert 1 in [e.n for e in recv1]  # type: ignore[attr-defined]
    assert 1 in [e.n for e in recv2], "late joiner missed the in-flight batch"  # type: ignore[attr-defined]


async def test_concurrent_remove_preserves_delivery(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    handler2: SimpleEventHandler2,
) -> None:
    """If one consumer is removed while the worker is reading, the remaining
    consumers still receive the batch."""
    cid = await service.create_stream("test")

    original_read = stream_reader.read_events
    removed = False

    async def read_then_remove(
        event_handler: Any,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        result = await original_read(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )
        nonlocal removed
        if result[0] and not removed:
            await multiplexer.remove_consumer(c2)
            removed = True
        return result

    stream_reader.read_events = read_then_remove  # type: ignore[method-assign]

    await multiplexer.start()
    c1 = await multiplexer.add_consumer(cid, handler)
    c2 = await multiplexer.add_consumer(cid, handler2)
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    try:
        recv1 = await collect_from_consumer(c1, timeout=5.0)
    finally:
        await multiplexer.stop()
        stream_reader.read_events = original_read  # type: ignore[method-assign]

    assert [e.n for e in recv1] == [1]  # type: ignore[attr-defined]


async def test_multiple_consumers_all_receive(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    handler2: SimpleEventHandler2,
) -> None:
    """Two distinct consumers on the same stream both receive every event."""
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.push_event(cid, json.dumps({"n": 2}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    c1 = await multiplexer.add_consumer(cid, handler)
    c2 = await multiplexer.add_consumer(cid, handler2)
    try:
        recv1 = await collect_from_consumer(c1, timeout=10.0)
        recv2 = await collect_from_consumer(c2, timeout=10.0)
    finally:
        await multiplexer.stop()

    assert [e.n for e in recv1] == [1, 2]  # type: ignore[attr-defined]
    assert [e.n for e in recv2] == [1, 2]  # type: ignore[attr-defined]


async def test_late_joiner_gets_full_history(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
    handler2: SimpleEventHandler2,
) -> None:
    """A consumer that joins AFTER the first batch has already been dispatched
    must still receive every event from the beginning (last_id='0').

    This is the 'Delta without start' bug: if a content_block_start event is
    in batch 0 and a second consumer only sees batch 1 onwards, the client
    receives content_block_delta with no preceding start.
    """
    cid = await service.create_stream("test")

    # Patch read_events so that after the first successful dispatch we can
    # confirm the broadcast already has batch 0, then add a second consumer.
    original_read = stream_reader.read_events
    second_consumer_box: dict[str, Any] = {}
    dispatch_count = 0

    async def read_after_first_dispatch(
        event_handler: Any,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        nonlocal dispatch_count
        result = await original_read(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )
        # After the first batch is returned (and will be dispatched), add c2.
        # At this point broadcast._batches is still empty (dispatch hasn't run),
        # but after dispatch it will have 1 batch — c2 must still see it.
        if result[0] and dispatch_count == 0 and "c2" not in second_consumer_box:
            dispatch_count += 1
            # Wait one event-loop tick so _dispatch runs and pushes batch 0.
            await asyncio.sleep(0)
            second_consumer_box["c2"] = await multiplexer.add_consumer(
                cid, handler2, last_id="0"
            )
        return result

    stream_reader.read_events = read_after_first_dispatch  # type: ignore[method-assign]

    await multiplexer.start()
    c1 = await multiplexer.add_consumer(cid, handler, last_id="0")

    await service.push_event(cid, json.dumps({"n": 1}))
    await service.push_event(cid, json.dumps({"n": 2}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    try:
        recv1 = await collect_from_consumer(c1)
        c2 = second_consumer_box["c2"]
        recv2 = await collect_from_consumer(c2)
    finally:
        await multiplexer.stop()
        stream_reader.read_events = original_read  # type: ignore[method-assign]

    assert [e.n for e in recv1] == [1, 2], "consumer 1 must receive all events"  # type: ignore[attr-defined]
    assert [e.n for e in recv2] == [1, 2], "late joiner must not miss batch 0"  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Terminal detection — events in the gap are drained
# ═══════════════════════════════════════════════════════════════════════════════


async def test_events_in_terminal_gap_drained(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    """Events published between the worker's empty read and the terminal-status
    check must be drained and delivered, not dropped."""
    cid = await service.create_stream("test")

    original_check = stream_reader.check_terminal_status

    async def injecting_check(
        correlation_id: str,
        event_handler: Any,
        cached_events: Any,
    ) -> bool:
        await service.push_event(correlation_id, json.dumps({"n": 1}))
        await service.push_event(correlation_id, json.dumps({"n": 2}))
        await service.update_stream_status(correlation_id, StreamStatus.COMPLETED)
        return True

    stream_reader.check_terminal_status = injecting_check  # type: ignore[method-assign]

    await multiplexer.start()
    consumer = await multiplexer.add_consumer(cid, handler)
    try:
        received = await collect_from_consumer(consumer, timeout=10.0)
    finally:
        await multiplexer.stop()
        stream_reader.check_terminal_status = original_check  # type: ignore[method-assign]

    assert [e.n for e in received] == [1, 2]  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Lifecycle — stop / restart leave no stale state
# ═══════════════════════════════════════════════════════════════════════════════


async def test_stop_clears_state_and_closes_consumers(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))

    await multiplexer.start()
    consumer = await multiplexer.add_consumer(cid, handler)

    # Wait until the worker has pushed at least one batch to the broadcast.
    async def get_one() -> None:
        async for _ in consumer.broadcast.read_from(cursor=consumer.cursor):
            break

    await asyncio.wait_for(get_one(), timeout=5.0)

    await multiplexer.stop()

    assert not multiplexer.consumers, "consumers dict must be empty after stop()"
    assert not multiplexer.states, "states dict must be empty after stop()"
    assert consumer.broadcast.is_closed, "broadcast must be closed after stop()"


async def test_restart_after_stop(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))
    await service.update_stream_status(cid, StreamStatus.COMPLETED)

    await multiplexer.start()
    await multiplexer.add_consumer(cid, handler)
    await multiplexer.stop()

    # Second stream on a fresh cid after restart.
    cid2 = await service.create_stream("test")
    await service.push_event(cid2, json.dumps({"n": 42}))
    await service.update_stream_status(cid2, StreamStatus.COMPLETED)

    await multiplexer.start()
    try:
        consumer = await multiplexer.add_consumer(cid2, handler)
        received = await collect_from_consumer(consumer, timeout=10.0)
    finally:
        await multiplexer.stop()

    assert [e.n for e in received] == [42]  # type: ignore[attr-defined]


async def test_stop_idempotent(
    multiplexer: StreamMultiplexer,
) -> None:
    await multiplexer.start()
    await multiplexer.stop()
    await multiplexer.stop()  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Performance — worker does not busy-poll
# ═══════════════════════════════════════════════════════════════════════════════


async def test_no_busy_poll(
    service: StreamService,
    stream_reader: StreamReader,
    multiplexer: StreamMultiplexer,
    handler: SimpleEventHandler,
) -> None:
    """With block_ms=DEFAULT_BLOCK_MS the worker parks on the backend; it must
    not spin.  In 300 ms we expect at most one read call."""
    cid = await service.create_stream("test")

    read_count = 0
    original_read = stream_reader.read_events

    async def counting_read(
        event_handler: Any,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[BaseModel], str]:
        nonlocal read_count
        read_count += 1
        return await original_read(
            event_handler=event_handler,
            correlation_id=correlation_id,
            last_id=last_id,
            count=count,
            block_ms=block_ms,
        )

    stream_reader.read_events = counting_read  # type: ignore[method-assign]

    await multiplexer.start()
    await multiplexer.add_consumer(cid, handler)
    await asyncio.sleep(0.3)
    await multiplexer.stop()
    stream_reader.read_events = original_read  # type: ignore[method-assign]

    assert read_count <= 1, (
        f"Worker called read_events {read_count} times in 300 ms "
        f"(expected ≤1 with block_ms={DEFAULT_BLOCK_MS})."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Backend block semantics — block_ms=None is non-blocking
# ═══════════════════════════════════════════════════════════════════════════════


async def test_block_ms_none_is_non_blocking(
    service: StreamService,
    stream_reader: StreamReader,
    handler: SimpleEventHandler,
) -> None:
    """block_ms=None means non-blocking on both backends.  This is what the
    direct path's final drain relies on, and it must never hang."""
    cid = await service.create_stream("test")
    await service.push_event(cid, json.dumps({"n": 1}))
    e1_id = await service.push_event(cid, json.dumps({"n": 2}))

    # Read past e1; no events remain.
    events, _ = await asyncio.wait_for(
        stream_reader.read_events(handler, cid, e1_id, None, None),
        timeout=2.0,
    )
    assert events == []


@pytest.mark.asyncio
async def test_concurrent_stream_creation_allows_only_one_replica() -> None:
    service = InMemoryStreamService()

    results = await asyncio.gather(
        *(
            service.create_stream("chat_completion", correlation_id="same-message")
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    assert results.count("same-message") == 1
    assert sum(isinstance(result, ValueError) for result in results) == 9
