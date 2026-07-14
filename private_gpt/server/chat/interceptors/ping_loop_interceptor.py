import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.events.models import Event, PingEvent

_DEFAULT_PING_INTERVAL = 15


class PingInterceptor(ChatResponseLoopInterceptor):
    def __init__(self, ping_interval: float = _DEFAULT_PING_INTERVAL) -> None:
        self._ping_interval = ping_interval
        self._ping_task: asyncio.Task[None] | None = None

    async def on_iteration_start(self, context: ChatInterceptorContext) -> None:
        emit_fn = context.emit_fn

        async def _ping_loop() -> None:
            with contextlib.suppress(asyncio.CancelledError):
                while True:
                    await asyncio.sleep(self._ping_interval)
                    emit_fn(PingEvent())

        self._ping_task = asyncio.create_task(_ping_loop())

    async def on_iteration_end(self, context: ChatInterceptorContext) -> None:
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task
        self._ping_task = None

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event | None:
        return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "PingInterceptor":
        # Return a new instance with the same logic but reset state
        return PingInterceptor(ping_interval=self._ping_interval)
