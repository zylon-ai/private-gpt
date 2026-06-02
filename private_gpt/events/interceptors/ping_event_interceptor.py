# ping_event_interceptor.py
import asyncio
import contextlib
from collections.abc import AsyncGenerator

from private_gpt.events.interceptors.base_event_interceptor import BaseEventInterceptor
from private_gpt.events.models import Event, PingEvent

_DEFAULT_PING_INTERVAL = 15


class PingEventInterceptor(BaseEventInterceptor):
    """Interceptor that emits ping events during idle periods."""

    def __init__(self, ping_interval: float | None = None):
        self.ping_interval = ping_interval or _DEFAULT_PING_INTERVAL

    async def intercept(
        self, gen: AsyncGenerator[Event, None]
    ) -> AsyncGenerator[Event, None]:
        async def coro() -> AsyncGenerator[Event, None]:

            event_queue: asyncio.Queue[Event | None] = asyncio.Queue()
            generator_done = asyncio.Event()
            exception_holder: list[BaseException] = []

            async def event_producer() -> None:
                try:
                    async for event in gen:
                        await event_queue.put(event)
                except Exception as e:
                    exception_holder.append(e)
                finally:
                    generator_done.set()
                    await event_queue.put(None)

            producer_task = asyncio.create_task(event_producer())

            try:
                while not generator_done.is_set() or not event_queue.empty():
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=self.ping_interval
                        )

                        if event is None:
                            break

                        yield event

                    except TimeoutError:
                        yield PingEvent()

                if exception_holder:
                    raise exception_holder[0]

            finally:
                if not producer_task.done():
                    producer_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await producer_task

        return coro()
