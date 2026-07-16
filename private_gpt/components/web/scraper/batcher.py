from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

BatchRunner = Callable[[list[str], int], Awaitable[list[str | Exception]]]


class _PendingScrape:
    __slots__ = ("future", "url")

    def __init__(self, url: str, future: asyncio.Future[str]) -> None:
        self.url = url
        self.future = future


class ScrapeBatcher:
    """Coalesces near-simultaneous scrape requests into one browser run.

    The first request opens a batch window of ``batch_wait_seconds``; requests
    arriving within it join the batch. The batch is dispatched early when it
    reaches ``batch_size``, otherwise when the window closes — so N pages
    requested at the same time share a single session/browser instead of
    launching N in parallel. ``batch_size=1`` dispatches immediately.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        batch_wait_seconds: float,
        run_batch: BatchRunner,
    ) -> None:
        self._batch_size = max(1, batch_size)
        self._wait = max(0.0, batch_wait_seconds)
        self._run_batch = run_batch
        self._pending: list[_PendingScrape] = []
        self._timeout_seconds = 0
        self._window: asyncio.Task[None] | None = None
        self._dispatches: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    async def submit(self, url: str, timeout_seconds: int) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        async with self._lock:
            if self._closed:
                raise RuntimeError("Scrape batcher is closed")
            self._pending.append(_PendingScrape(url, future))
            self._timeout_seconds = max(self._timeout_seconds, timeout_seconds)
            if len(self._pending) >= self._batch_size:
                self._flush_locked()
            elif self._window is None:
                self._window = asyncio.create_task(self._close_window())
        return await future

    def _flush_locked(self) -> None:
        """Dispatch the pending batch. Caller must hold the lock."""
        if self._window is not None:
            self._window.cancel()
            self._window = None
        batch, self._pending = self._pending, []
        timeout_seconds, self._timeout_seconds = self._timeout_seconds, 0
        if batch:
            task = asyncio.create_task(self._dispatch(batch, timeout_seconds))
            self._dispatches.add(task)
            task.add_done_callback(self._dispatches.discard)

    async def _close_window(self) -> None:
        await asyncio.sleep(self._wait)
        async with self._lock:
            self._window = None
            self._flush_locked()

    async def _dispatch(
        self, batch: list[_PendingScrape], timeout_seconds: int
    ) -> None:
        try:
            results = await self._run_batch([p.url for p in batch], timeout_seconds)
        except BaseException as exc:
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return
        for pending, result in zip(batch, results, strict=True):
            if pending.future.done():
                continue
            if isinstance(result, Exception):
                pending.future.set_exception(result)
            else:
                pending.future.set_result(result)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._window is not None:
                self._window.cancel()
                self._window = None
            pending, self._pending = self._pending, []
        for p in pending:
            if not p.future.done():
                p.future.set_exception(RuntimeError("Scrape batcher is closed"))
        if self._dispatches:
            await asyncio.gather(*self._dispatches, return_exceptions=True)
