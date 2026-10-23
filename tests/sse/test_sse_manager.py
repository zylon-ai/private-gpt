import asyncio
import queue
import time
from typing import TYPE_CHECKING

import pytest

from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    FatalError,
    PingEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from private_gpt.events.sse.sse_manager import SSEStreamManager
from private_gpt.events.sse.sse_producer import SSEProducer

if TYPE_CHECKING:
    from private_gpt.events.models import Event


@pytest.fixture
def manager() -> SSEStreamManager:
    return SSEStreamManager()


@pytest.fixture
def producer(manager: SSEStreamManager) -> SSEProducer:
    return SSEProducer(manager=manager, model_name="test-model")


def test_manager_initialization() -> None:
    manager = SSEStreamManager()
    assert isinstance(manager._queue, queue.Queue)
    assert isinstance(manager._aqueue, asyncio.Queue)
    assert manager._sync_thread is None
    assert manager._async_thread is None
    assert manager._lock is not None


def test_send_event(manager: SSEStreamManager) -> None:
    event = PingEvent()
    manager.send_event(event)

    # Get event from queue
    queued_event = manager._queue.get_nowait()
    assert queued_event == event

    # Check async queue as well
    asyncio.set_event_loop(asyncio.new_event_loop())

    async def check_async_queue() -> None:
        queued_event = await manager._aqueue.get()
        assert queued_event == event

    asyncio.run(check_async_queue())


def test_message_stream(manager: SSEStreamManager, producer: SSEProducer) -> None:
    collected_events = []

    with producer.message_stream() as message_start:
        collected_events.append(manager._queue.get_nowait())
        assert message_start.message.id is not None
        assert message_start.message.model == "test-model"

    # Check more events were added
    while not manager._queue.empty():
        event = manager._queue.get_nowait()
        if event is None:
            break
        collected_events.append(event)

    # Check we have message start and stop events
    assert len(collected_events) == 2
    assert isinstance(collected_events[0], RawMessageStartEvent)
    assert isinstance(collected_events[1], RawMessageStopEvent)


def test_message_stream_with_exception(
    manager: SSEStreamManager, producer: SSEProducer
) -> None:
    collected_events = []

    try:
        with producer.message_stream():
            collected_events.append(manager._queue.get_nowait())  # Message start
            raise ValueError("Test error")
    except ValueError:
        pass

    # Collect all events
    while not manager._queue.empty():
        event = manager._queue.get_nowait()
        if event is None:
            break
        collected_events.append(event)

    # Check error and done events are sent
    assert len(collected_events) == 2
    assert isinstance(collected_events[0], RawMessageStartEvent)
    assert isinstance(collected_events[1], FatalError)
    assert collected_events[1].type == "error"
    assert (
        collected_events[1].error.type
        == Errors._EXCEPTION_TO_ERROR[ValueError].error_type
    )


def test_sync_stream(producer: SSEProducer) -> None:
    def handler() -> None:
        producer.send_ping()
        producer.send_ping()
        time.sleep(0.1)
        producer.send_ping()

    events_processed: list[Event] = []
    for event in producer.manager.stream(handler):
        events_processed.append(event)

    assert len(events_processed) == 3
    assert all(isinstance(e, PingEvent) for e in events_processed)


@pytest.mark.asyncio
async def test_async_stream(producer: SSEProducer) -> None:
    async def handler() -> None:
        producer.send_ping()
        producer.send_ping()
        await asyncio.sleep(0.1)
        producer.send_ping()

    events_processed: list[Event] = []
    async for event in producer.manager.astream(handler):
        events_processed.append(event)

    assert len(events_processed) == 3
    assert all(isinstance(e, PingEvent) for e in events_processed)
