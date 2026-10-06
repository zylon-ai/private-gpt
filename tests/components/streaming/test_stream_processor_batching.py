from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from private_gpt.components.streaming.providers.in_memory_stream_service import (
    InMemoryStreamService,
)
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.stream.stream_processor import StreamProcessor
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _SlowWrites(InMemoryStreamService):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []
        self.status_after_events: list[tuple[StreamStatus, int]] = []

    async def push_events(self, correlation_id: str, event_datas: list[str]) -> str:
        self.batches.append(len(event_datas))
        await asyncio.sleep(0.01)  # a round trip during which tokens pile up
        return await super().push_events(correlation_id, event_datas)

    async def update_stream_status(self, correlation_id, status, **kwargs):  # type: ignore[no-untyped-def]
        written = sum(self.batches)
        self.status_after_events.append((status, written))
        return await super().update_stream_status(correlation_id, status, **kwargs)


def _processor(service: InMemoryStreamService) -> StreamProcessor:
    component = MagicMock()
    component.stream = service
    task_manager = MagicMock()
    task_manager.is_cancelled = MagicMock(return_value=False)
    return StreamProcessor(stream_component=component, task_manager=task_manager)


async def _events(n: int, gap: float) -> AsyncGenerator[object, None]:
    yield RawMessageStartEvent.from_defaults()
    start = RawContentBlockStartEvent.model_validate(
        {
            "type": "content_block_start",
            "block_id": "b",
            "content_block": {"type": "text", "text": ""},
        }
    )
    yield start
    for _ in range(n):
        yield start  # payload content is irrelevant here; order and count are not
        if gap:
            await asyncio.sleep(gap)
        else:
            await asyncio.sleep(0)
    yield RawMessageStopEvent.from_defaults()


async def _read_all(service: InMemoryStreamService, cid: str) -> list[str]:
    events, _ = await service.read_events(cid, "0")
    return events


@pytest.mark.anyio
async def test_burst_is_written_in_few_batches_in_order() -> None:
    service = _SlowWrites()
    await service.create_stream("chat_completion", correlation_id="c")
    handler = StreamingEventHandler()
    await _processor(service).process_stream(
        "c", "chat_completion", _events(200, 0), handler
    )
    written = await _read_all(service, "c")
    assert len(written) == 203
    assert sum(service.batches) == 203
    assert len(service.batches) < 50  # far fewer round trips than events
    assert '"message_start"' in written[0]
    assert '"message_stop"' in written[-1]
    metadata = await service.get_stream_metadata("c")
    assert metadata is not None
    assert metadata.status == StreamStatus.COMPLETED


@pytest.mark.anyio
async def test_slow_stream_writes_each_event_immediately() -> None:
    service = _SlowWrites()
    await service.create_stream("chat_completion", correlation_id="c")
    await _processor(service).process_stream(
        "c", "chat_completion", _events(5, 0.03), StreamingEventHandler()
    )
    # With gaps longer than a write, nothing waits to be batched.
    assert all(size == 1 for size in service.batches[2:-1])


@pytest.mark.anyio
async def test_processing_status_is_set_after_earlier_events_are_written() -> None:
    service = _SlowWrites()
    await service.create_stream("chat_completion", correlation_id="c")
    await _processor(service).process_stream(
        "c", "chat_completion", _events(3, 0), StreamingEventHandler()
    )
    processing = [
        n
        for status, n in service.status_after_events
        if status == StreamStatus.PROCESSING
    ]
    assert processing == [1]  # message_start written before PROCESSING


@pytest.mark.anyio
async def test_write_failure_is_reported_as_stream_error() -> None:
    service = _SlowWrites()
    await service.create_stream("chat_completion", correlation_id="c")
    calls = 0
    original = service.push_events

    async def failing(correlation_id: str, event_datas: list[str]) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("redis down")
        return await original(correlation_id, event_datas)

    service.push_events = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="redis down"):
        await _processor(service).process_stream(
            "c", "chat_completion", _events(50, 0), StreamingEventHandler()
        )
    metadata = await service.get_stream_metadata("c")
    assert metadata is not None
    assert metadata.status == StreamStatus.ERROR
