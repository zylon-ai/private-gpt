"""Async priority queue with local (in-process) concurrency control."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from private_gpt.components.concurrency.semaphore_manager import (
    QueueShutdownError,
    SemaphoreManager,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class MemorySemaphoreManager(SemaphoreManager):
    def __init__(
        self,
        max_concurrency: int | None = None,
        queue_key: str | None = None,
    ) -> None:
        self.max_concurrency: int = max_concurrency or 2
        self.queue_key = queue_key or "local"

        self._tasks: dict[
            str,
            tuple[Callable[..., Awaitable[Any]], dict[str, Any], asyncio.Future[Any]],
        ] = {}
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, str]
        ] = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task[None]] = []
        self._shutdown = asyncio.Event()
        self._started = False
        self._counter = 0

    async def __aenter__(self) -> MemorySemaphoreManager:
        await self.start_processor()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _execute_task(
        self,
        request_id: str,
        priority: int,
    ) -> None:
        task_info = self._tasks.pop(request_id, None)
        if not task_info:
            logger.error("Task %s not found in local registry", request_id)
            return

        func, kwargs, future = task_info
        try:
            logger.debug(
                "Executing local task %s (priority=%d)",
                request_id,
                priority,
            )
            result = await func(**kwargs)
            if not future.done():
                future.set_result(result)
        except Exception as e:
            logger.error("Local task %s failed: %s", request_id, e, exc_info=True)
            if not future.done():
                future.set_exception(e)

    async def _worker_loop(self, worker_idx: int) -> None:
        logger.debug(
            "Local worker %d started for queue: %s", worker_idx, self.queue_key
        )
        while not self._shutdown.is_set():
            try:
                priority, _, request_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.5,
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._execute_task(request_id, priority)
            except Exception as e:
                logger.error("Local worker error: %s", e, exc_info=True)
            finally:
                self._queue.task_done()

        logger.debug(
            "Local worker %d stopped for queue: %s", worker_idx, self.queue_key
        )

    async def start_processor(self) -> None:
        if self._started:
            return

        self._started = True
        self._shutdown.clear()
        self._workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.max_concurrency)
        ]
        logger.info(
            "Started local workers for queue %s (max=%d)",
            self.queue_key,
            self.max_concurrency,
        )

    async def execute(
        self,
        task_func: Callable[..., Awaitable[Any]],
        priority: int = 0,
        **kwargs: Any,
    ) -> Any:
        if self._shutdown.is_set():
            raise QueueShutdownError("Queue is shutting down")

        await self.start_processor()
        if self._shutdown.is_set():
            raise QueueShutdownError("Queue is shutting down")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[Any] = asyncio.Future()
        self._tasks[request_id] = (task_func, kwargs, future)

        self._counter += 1
        # Lower value == higher priority
        self._queue.put_nowait((priority, self._counter, request_id))
        logger.debug("Enqueued local task %s (priority=%d)", request_id, priority)

        try:
            return await future
        except asyncio.CancelledError:
            self._tasks.pop(request_id, None)
            raise

    async def close(self) -> None:
        logger.info("Closing local queue: %s", self.queue_key)
        self._shutdown.set()

        for _, (_, _, future) in list(self._tasks.items()):
            if not future.done():
                future.cancel()
        self._tasks.clear()

        for worker in self._workers:
            worker.cancel()
            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=5.0)
        self._workers = []

        while not self._queue.empty():
            with suppress(Exception):
                self._queue.get_nowait()
                self._queue.task_done()

        logger.info("Local queue closed: %s", self.queue_key)
