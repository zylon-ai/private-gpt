import asyncio
from collections.abc import AsyncGenerator

import pytest

from private_gpt.events.interceptors.ping_event_interceptor import (
    PingEventInterceptor,
)
from private_gpt.events.models import Event, PingEvent, RawMessageStopEvent


@pytest.mark.anyio
async def test_ping_is_emitted_while_listener_waits_for_resume() -> None:
    resume = asyncio.Event()

    async def paused_stream() -> AsyncGenerator[Event, None]:
        await resume.wait()
        yield RawMessageStopEvent()

    stream = await PingEventInterceptor(ping_interval=0.01).intercept(paused_stream())

    assert isinstance(await anext(stream), PingEvent)

    resume.set()
    assert isinstance(await anext(stream), RawMessageStopEvent)


@pytest.mark.anyio
async def test_closing_listener_closes_paused_stream() -> None:
    generator_closed = asyncio.Event()

    async def paused_stream() -> AsyncGenerator[Event, None]:
        try:
            await asyncio.Event().wait()
            if False:
                yield PingEvent()
        finally:
            generator_closed.set()

    stream = await PingEventInterceptor(ping_interval=0.01).intercept(paused_stream())
    assert isinstance(await anext(stream), PingEvent)

    await stream.aclose()

    assert generator_closed.is_set()


@pytest.mark.anyio
async def test_events_pass_through_in_order_without_pings() -> None:
    async def fast_stream() -> AsyncGenerator[Event, None]:
        for _ in range(50):
            yield RawMessageStopEvent()
            await asyncio.sleep(0)

    stream = await PingEventInterceptor(ping_interval=5).intercept(fast_stream())
    events = [event async for event in stream]

    assert len(events) == 50
    assert all(isinstance(event, RawMessageStopEvent) for event in events)


@pytest.mark.anyio
async def test_producer_exception_is_raised_after_buffered_events() -> None:
    async def failing_stream() -> AsyncGenerator[Event, None]:
        yield RawMessageStopEvent()
        raise RuntimeError("boom")

    stream = await PingEventInterceptor(ping_interval=5).intercept(failing_stream())

    assert isinstance(await anext(stream), RawMessageStopEvent)
    with pytest.raises(RuntimeError, match="boom"):
        await anext(stream)


@pytest.mark.anyio
async def test_pings_repeat_while_idle() -> None:
    resume = asyncio.Event()

    async def paused_stream() -> AsyncGenerator[Event, None]:
        await resume.wait()
        yield RawMessageStopEvent()

    stream = await PingEventInterceptor(ping_interval=0.01).intercept(paused_stream())

    assert isinstance(await anext(stream), PingEvent)
    assert isinstance(await anext(stream), PingEvent)
    resume.set()
    rest = [event async for event in stream]
    assert isinstance(rest[-1], RawMessageStopEvent)


@pytest.mark.anyio
async def test_does_not_create_a_task_per_event() -> None:
    async def fast_stream() -> AsyncGenerator[Event, None]:
        for _ in range(200):
            yield RawMessageStopEvent()
            await asyncio.sleep(0)

    loop = asyncio.get_running_loop()
    created = 0
    original = loop.create_task

    def counting_create_task(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal created
        created += 1
        return original(*args, **kwargs)

    loop.create_task = counting_create_task  # type: ignore[method-assign]
    try:
        stream = await PingEventInterceptor(ping_interval=5).intercept(fast_stream())
        events = [event async for event in stream]
    finally:
        loop.create_task = original  # type: ignore[method-assign]

    assert len(events) == 200
    assert created <= 2  # the producer task (and nothing per event)
