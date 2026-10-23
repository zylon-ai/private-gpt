import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, TypeVar

from starlette.requests import Request

T = TypeVar("T")


async def cancel_on_http_disconnect(
    request: Request,
    operation: Coroutine[Any, Any, T],
    poll_interval: float = 0.1,
) -> T:
    task = asyncio.create_task(operation)
    try:
        await asyncio.sleep(0)
        while not task.done():
            if await request.is_disconnected():
                raise asyncio.CancelledError("HTTP request disconnected")
            await asyncio.sleep(poll_interval)
        return await task
    except asyncio.CancelledError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        raise
