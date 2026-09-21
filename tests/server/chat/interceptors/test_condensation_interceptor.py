import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
    CondenseResponse,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    TLDRBlock,
)
from private_gpt.server.chat.interceptors.condensation_interceptor import (
    _condensation_producer,
    _CondensationResult,
    _consume_and_emit_with_min_duration,
    _OpenBlockTracker,
)

_HISTORY = [ChatMessage(role=MessageRole.USER, content="final question")]


def _tldr_block(side: str) -> TextBlock:
    return TextBlock(
        text="summary",
        metadata={"type": "tldr", "role": "assistant", "tldr_side": side},
    )


def _assert_well_formed(events: list[Event]) -> None:
    """Every block opens once, closes once, and closes only after it opened.

    Guards both directions: a start without its stop strands the TLDR block
    "in progress" on the client, while a duplicate or orphan stop corrupts the
    client's block state just as badly.
    """
    starts = [e.block_id for e in events if isinstance(e, RawContentBlockStartEvent)]
    stops = [e.block_id for e in events if isinstance(e, RawContentBlockStopEvent)]

    assert len(starts) == len(set(starts)), f"duplicate start events: {starts}"
    assert len(stops) == len(set(stops)), f"duplicate stop events: {stops}"
    assert not set(stops) - set(starts), "stop event for a block that never started"
    assert not set(starts) - set(stops), "block left open"

    open_ids: set[str] = set()
    for event in events:
        if isinstance(event, RawContentBlockStartEvent):
            open_ids.add(event.block_id)
        elif isinstance(event, RawContentBlockStopEvent):
            assert event.block_id in open_ids, "stop emitted before its start"


async def _run_producer(
    generator: AsyncIterator[CondenseResponse],
    *,
    min_duration: float | None = 0.01,
    emit_fn: Callable[[Event], None] | None = None,
) -> tuple[list[Event], BaseException | None]:
    queue: asyncio.Queue[Event | object | None] = asyncio.Queue()
    emitted: list[Event] = []
    emit = emit_fn if emit_fn is not None else emitted.append
    producer = asyncio.create_task(
        _condensation_producer(queue, _CondensationResult(), generator)
    )
    tracker = _OpenBlockTracker()

    error: BaseException | None = None
    try:
        await _consume_and_emit_with_min_duration(
            emit, queue, min_duration=min_duration, tracker=tracker
        )
        with contextlib.suppress(Exception):
            await producer
    except BaseException as exc:
        error = exc
        if not producer.done():
            producer.cancel()
        with contextlib.suppress(BaseException):
            await producer
        # Same emit callable as the consumer used, mirroring intercept(), which
        # passes context.emit_event to both.
        tracker.close_pending(emit)

    if producer.done() and not producer.cancelled():
        error = error or producer.exception()

    return emitted, error


async def _start_then_raise(exc: BaseException) -> AsyncIterator[CondenseResponse]:
    # A delta is emitted before failing so the start event actually reaches the
    # client; otherwise the min_duration suppression hides the leak entirely.
    yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
    yield CondenseResponse(
        is_condensed=True,
        chat_history=_HISTORY,
        condense_blocks=[_tldr_block("left")],
    )
    await asyncio.sleep(0)
    raise exc


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("summarizer exploded"), TimeoutError("condensation timed out")],
    ids=["strategy_error", "timeout"],
)
async def test_open_block_is_closed_when_condensation_fails(
    exc: BaseException,
) -> None:
    """A failure after the start event must still close the TLDR block.

    Regression: the stop events were emitted inside the ``try`` body, so any
    exception skipped them and left the block streaming "in progress" on the
    client until the stream itself timed out.
    """
    emitted, error = await _run_producer(_start_then_raise(exc))

    assert error is not None
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_open_block_is_closed_when_failure_happens_after_a_delta() -> None:
    async def generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left")],
        )
        raise RuntimeError("exploded after emitting a delta")

    emitted, error = await _run_producer(generator())

    assert isinstance(error, RuntimeError)
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_open_block_is_closed_when_consumer_is_cancelled() -> None:
    """Client disconnect mid-condensation must not strand an open block."""

    async def slow_generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        await asyncio.sleep(30)
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left")],
        )

    queue: asyncio.Queue[Event | object | None] = asyncio.Queue()
    emitted: list[Event] = []
    producer = asyncio.create_task(
        _condensation_producer(queue, _CondensationResult(), slow_generator())
    )

    tracker = _OpenBlockTracker()
    consumer = asyncio.create_task(
        _consume_and_emit_with_min_duration(
            emitted.append, queue, min_duration=0.01, tracker=tracker
        )
    )
    await asyncio.sleep(0.1)
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await consumer

    producer.cancel()
    with contextlib.suppress(BaseException):
        await producer
    tracker.close_pending(emitted.append)

    assert any(isinstance(event, RawContentBlockStartEvent) for event in emitted)
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_happy_path_emits_matched_start_and_stop_per_side() -> None:
    async def generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        await asyncio.sleep(0)
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left"), _tldr_block("right")],
        )

    emitted, error = await _run_producer(generator())

    assert error is None
    _assert_well_formed(emitted)
    assert sum(isinstance(e, RawContentBlockStartEvent) for e in emitted) == 2
    assert sum(isinstance(e, RawContentBlockStopEvent) for e in emitted) == 2


@pytest.mark.asyncio
async def test_min_duration_suppression_leaves_nothing_open() -> None:
    """A run suppressed by min_duration must emit neither start nor stop."""

    async def fast_generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        yield CondenseResponse(
            is_condensed=True, chat_history=_HISTORY, condense_blocks=[]
        )

    emitted, error = await _run_producer(fast_generator(), min_duration=0.2)

    assert error is None
    assert emitted == []


@pytest.mark.asyncio
async def test_suppressed_block_is_not_stopped_on_cancellation() -> None:
    """A block the client never saw must not receive a stop event.

    Regression: the close path replayed stop events straight off the queue, so a
    start still suppressed by ``min_duration`` produced an orphan
    ``content_block_stop`` for a block the client had never been told about.
    """

    async def generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left")],
        )

    queue: asyncio.Queue[Event | object | None] = asyncio.Queue()
    emitted: list[Event] = []
    producer = asyncio.create_task(
        _condensation_producer(queue, _CondensationResult(), generator())
    )

    tracker = _OpenBlockTracker()
    # A long min_duration keeps the consumer asleep, so nothing has been emitted
    # yet when the cancellation lands.
    consumer = asyncio.create_task(
        _consume_and_emit_with_min_duration(
            emitted.append, queue, min_duration=5, tracker=tracker
        )
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with contextlib.suppress(BaseException):
        await consumer

    if not producer.done():
        producer.cancel()
    with contextlib.suppress(BaseException):
        await producer
    tracker.close_pending(emitted.append)

    assert emitted == []
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_block_is_closed_when_emit_fails_midway() -> None:
    """A failing emit must not strand the block it was midway through.

    Regression: the consumer drained the queue into a local buffer before
    emitting, so an ``emit_fn`` that raised took the pending stop event down with
    it and left the queue empty for the close path to find.
    """
    emitted: list[Event] = []
    calls = {"n": 0}

    def flaky_emit(event: Event) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            # Fails right after the start event has gone out to the client.
            raise RuntimeError("client connection gone")
        emitted.append(event)

    async def generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left")],
        )

    _, error = await _run_producer(generator(), emit_fn=flaky_emit)

    assert isinstance(error, RuntimeError)
    assert any(isinstance(event, RawContentBlockStartEvent) for event in emitted)
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_close_pending_is_idempotent() -> None:
    """Closing twice must not emit a duplicate stop for the same block."""
    emitted: list[Event] = []
    tracker = _OpenBlockTracker()
    start = RawContentBlockStartEvent(
        block_id="block_test",
        content_block=TLDRBlock(content=[], tldr_side="left"),
    )

    tracker.emit(emitted.append, start)
    tracker.close_pending(emitted.append)
    tracker.close_pending(emitted.append)

    assert sum(isinstance(e, RawContentBlockStopEvent) for e in emitted) == 1
    _assert_well_formed(emitted)


@pytest.mark.asyncio
async def test_block_whose_start_failed_to_emit_is_not_stopped() -> None:
    """If the start never reached the client, no stop may be sent for it.

    The tracker records a block only after ``emit_fn`` returns, so a start that
    failed on the way out is never considered open.
    """
    emitted: list[Event] = []

    def reject_start(event: Event) -> None:
        if isinstance(event, RawContentBlockStartEvent):
            raise RuntimeError("client connection gone")
        emitted.append(event)

    async def generator() -> AsyncIterator[CondenseResponse]:
        yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])
        yield CondenseResponse(
            is_condensed=True,
            chat_history=_HISTORY,
            condense_blocks=[_tldr_block("left")],
        )

    _, error = await _run_producer(generator(), emit_fn=reject_start)

    assert isinstance(error, RuntimeError)
    assert emitted == []
    _assert_well_formed(emitted)
