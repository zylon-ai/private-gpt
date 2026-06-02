import asyncio
import contextlib
import logging
from collections.abc import Coroutine
from contextvars import copy_context
from typing import Any

from injector import inject, singleton

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class TaskManager:
    """Manages asyncio tasks with concurrency control and cancellation support."""

    @inject
    def __init__(self, settings: Settings) -> None:
        max_concurrent_tasks = settings.chat.maximum_concurrent_requests or None
        self._semaphore = (
            asyncio.Semaphore(max_concurrent_tasks)
            if max_concurrent_tasks is not None
            else None
        )
        self._active_tasks: dict[str, asyncio.Task[Any | None]] = {}
        self._cancellation_tokens: dict[str, asyncio.Event] = {}

    async def create_task(
        self,
        correlation_id: str,
        coro: Coroutine[Any, Any, None],
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        """Create and register a new task with concurrency control."""
        if correlation_id in self._active_tasks:
            logger.warning(f"Task {correlation_id} already exists")
            return self._active_tasks[correlation_id]

        async def wrapped_coro() -> None:
            try:
                if self._semaphore is not None:
                    await self._semaphore.acquire()
                    logger.debug(
                        f"Acquired slot for {correlation_id} "
                        f"(active: {len(self._active_tasks)})"
                    )

                await coro
            finally:
                if self._semaphore is not None:
                    self._semaphore.release()
                    logger.debug(f"Released slot for {correlation_id}")

        cancellation_token = asyncio.Event()
        self._cancellation_tokens[correlation_id] = cancellation_token

        ctx = copy_context()
        task = asyncio.create_task(wrapped_coro(), name=name, context=ctx)
        self._active_tasks[correlation_id] = task
        task.add_done_callback(lambda _: self._cleanup_task(correlation_id))

        return task

    def get_task(self, correlation_id: str) -> asyncio.Task[Any | None] | None:
        """Get task by correlation ID."""
        return self._active_tasks.get(correlation_id)

    def get_cancellation_token(self, correlation_id: str) -> asyncio.Event | None:
        """Get cancellation token by correlation ID."""
        return self._cancellation_tokens.get(correlation_id)

    def is_cancelled(self, correlation_id: str) -> bool:
        """Check if task is cancelled."""
        token = self._cancellation_tokens.get(correlation_id)
        return token is not None and token.is_set()

    async def cancel_task(self, correlation_id: str, timeout: float = 2.0) -> bool:
        """Cancel a task and wait for completion."""
        if correlation_id in self._cancellation_tokens:
            self._cancellation_tokens[correlation_id].set()

        task = self._active_tasks.get(correlation_id)
        if task is None:
            return False

        task.cancel()

        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout)

        return True

    def _cleanup_task(self, correlation_id: str) -> None:
        """Clean up task references."""
        self._active_tasks.pop(correlation_id, None)
        self._cancellation_tokens.pop(correlation_id, None)

    async def cancel_all_tasks(self) -> None:
        """Cancel all active tasks."""
        tasks = list(self._active_tasks.keys())
        await asyncio.gather(
            *[self.cancel_task(task_id) for task_id in tasks],
            return_exceptions=True,
        )

    def get_active_tasks(self) -> dict[str, asyncio.Task[Any]]:
        """Get all active tasks."""
        return self._active_tasks.copy()

    def get_active_count(self) -> int:
        """Get number of active tasks."""
        return len(self._active_tasks)
