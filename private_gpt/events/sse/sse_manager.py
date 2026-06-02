import asyncio
import contextlib
import queue
from collections.abc import AsyncIterator, Iterator
from threading import Lock, Thread
from typing import Any, Protocol

from private_gpt.events.models import (
    Event,
)


class SSEEventProducer(Protocol):
    def __call__(self) -> None:
        pass


class AsyncSSEEventProducer(Protocol):
    async def __call__(self) -> None:
        pass


class SSEStreamManager:
    def __init__(self) -> None:
        self._queue = queue.Queue[Event | None]()
        self._aqueue = asyncio.Queue[Event | None]()

        self._sync_thread: Thread | None = None
        self._async_thread: Thread | None = None
        self._lock = Lock()

    def send_event(self, event: Event | None) -> None:
        # Send event
        with self._lock:
            self._queue.put_nowait(event)
            self._aqueue.put_nowait(event)

    def __iter__(self) -> Iterator[Event]:
        while True:
            try:
                event = self._queue.get()
                if event is None:
                    break
                yield event
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            try:
                event = await self._aqueue.get()
                if event is None:
                    break
                yield event
                self._aqueue.task_done()
            except asyncio.CancelledError:
                break

    def __enter__(self) -> "SSEStreamManager":
        self._queue = queue.Queue[Event | None]()
        self._aqueue = asyncio.Queue[Event | None]()
        return self

    def __exit__(self, exc_type: Exception, exc_value: str, traceback: str) -> None:
        self.close()

    def stream(self, producer: SSEEventProducer) -> Iterator[Event]:
        def run_producer() -> None:
            try:
                producer()
            finally:
                # Signal end of stream
                self._queue.put(None)

        try:
            self._sync_thread = Thread(target=run_producer, daemon=True)
            self._sync_thread.start()

            for event in self:
                if not event:
                    break
                yield event
        finally:
            if self._sync_thread:
                self._sync_thread.join(timeout=1)

    async def astream(self, producer: AsyncSSEEventProducer) -> AsyncIterator[Event]:
        async def run_producer() -> None:
            try:
                await producer()
            finally:
                # Signal end of stream
                await self._aqueue.put(None)

        task: asyncio.Task[Any] | None = None
        try:
            task = asyncio.create_task(run_producer())
            async for event in self:
                if not event:
                    break
                yield event
        finally:
            if task:
                task.cancel()

    def close(self) -> None:
        """Close the stream manager and clean up resources."""
        with self._lock:
            # Signal end of streams
            self._queue.put(None)
            with contextlib.suppress(RuntimeError, ValueError):
                self._aqueue.task_done()

            # Join threads if they exist
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=1.0)
                self._sync_thread = None

            if self._async_thread and self._async_thread.is_alive():
                self._async_thread.join(timeout=1.0)
                self._async_thread = None

    async def aclose(self) -> None:
        """Close the stream manager asynchronously."""
        with self._lock:
            # Signal end of streams
            self._queue.put(None)
            await self._aqueue.put(None)

            # Join threads if they exist
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=1.0)
                self._sync_thread = None

            if self._async_thread and self._async_thread.is_alive():
                self._async_thread.join(timeout=1.0)
                self._async_thread = None
