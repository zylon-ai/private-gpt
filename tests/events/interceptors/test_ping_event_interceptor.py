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
