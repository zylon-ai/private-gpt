# ping_event_interceptor.py
import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncGenerator

from private_gpt.events.interceptors.base_event_interceptor import BaseEventInterceptor
from private_gpt.events.models import Event, PingEvent

_DEFAULT_PING_INTERVAL = 15

_END = object()
_PING = object()


class PingEventInterceptor(BaseEventInterceptor):
    """Interceptor that emits ping events during idle periods.

    Events are handed over through a deque and one waiter future; a single
    loop timer marks idle periods. The previous version wrapped every
    ``queue.get`` in ``asyncio.wait_for``, which on Python 3.11 creates a Task
    per streamed event (~10% of the API event loop under load).
    """

    def __init__(self, ping_interval: float | None = None):
        self.ping_interval = ping_interval or _DEFAULT_PING_INTERVAL

    async def intercept(
        self, gen: AsyncGenerator[Event, None]
    ) -> AsyncGenerator[Event, None]:
        async def coro() -> AsyncGenerator[Event, None]:
            loop = asyncio.get_running_loop()
            items: deque[object] = deque()
            waiter: asyncio.Future[None] | None = None
            exception_holder: list[BaseException] = []

            def wake() -> None:
                if waiter is not None and not waiter.done():
                    waiter.set_result(None)

            def push(item: object) -> None:
                items.append(item)
                wake()

            async def event_producer() -> None:
                try:
                    async for event in gen:
                        push(event)
                except Exception as e:
                    exception_holder.append(e)
                finally:
                    push(_END)

            producer_task = asyncio.create_task(event_producer())
            timer: asyncio.TimerHandle | None = None

            def on_idle() -> None:
                push(_PING)

            try:
                while True:
                    if not items:
                        # Arm the idle timer only while waiting; any event
                        # cancels it, so pings follow ping_interval of silence.
                        timer = loop.call_later(self.ping_interval, on_idle)
                        waiter = loop.create_future()
                        try:
                            await waiter
                        finally:
                            waiter = None
                            timer.cancel()
                            timer = None
                    item = items.popleft()
                    if item is _END:
                        break
                    if item is _PING:
                        yield PingEvent()
                        continue
                    yield item  # type: ignore[misc]

                if exception_holder:
                    raise exception_holder[0]

            finally:
                if timer is not None:
                    timer.cancel()
                if not producer_task.done():
                    producer_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await producer_task

        return coro()
