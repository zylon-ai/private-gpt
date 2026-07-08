from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from injector import inject, singleton
from pydantic import BaseModel, ConfigDict

from private_gpt.components.tools.tool_names import (
    BASH_TOOL_NAME,
    DATABASE_QUERY_TOOL_NAME,
    SEMANTIC_SEARCH_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_TOOL_COMPLEXITY: dict[str, float] = {
    SEMANTIC_SEARCH_TOOL_NAME: 0.9,
    WEB_SEARCH_TOOL_NAME: 0.9,
    BASH_TOOL_NAME: 0.9,
    WEB_FETCH_TOOL_NAME: 0.8,
    DATABASE_QUERY_TOOL_NAME: 0.8,
}
_DEFAULT_COMPLEXITY = 0.3


def _chat_urgency(chat_priority: int | None) -> float:
    """Map system.priority to [0, 1]; higher = more urgent.

    DEFAULT (0) and None have the worst urgency (no priority set).
    REAL_TIME (1) is the highest urgency.
    NO_PRIORITY (2) is explicitly the lowest.
    """
    if chat_priority is None or chat_priority == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (chat_priority - 1)))


class BaseToolScheduler(ABC):
    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        ...


class ImmediateToolScheduler(BaseToolScheduler):
    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        return await func()


class _PendingCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float
    counter: int
    entry_id: str
    func: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]

    def __lt__(self, other: _PendingCall) -> bool:
        return (self.score, self.counter) < (other.score, other.counter)


@singleton
class QueuedToolScheduler(BaseToolScheduler):
    @inject
    def __init__(self, settings: Settings) -> None:
        ts = settings.tool_scheduler
        self._max_concurrent: int = ts.max_concurrent_tools
        self._w_priority: float = ts.weights.chat_priority
        self._w_complexity: float = ts.weights.complexity

        self._queue: asyncio.PriorityQueue[_PendingCall] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._counter: int = 0
        self._started: bool = False

    def _get_queue(self) -> asyncio.PriorityQueue[_PendingCall]:
        if self._queue is None:
            self._queue = asyncio.PriorityQueue()
        return self._queue

    def _score(self, tool_name: str, chat_priority: int | None) -> float:
        urgency = _chat_urgency(chat_priority)
        complexity = _TOOL_COMPLEXITY.get(tool_name, _DEFAULT_COMPLEXITY)
        denom = self._w_priority + self._w_complexity
        combined = (
            (self._w_priority * urgency + self._w_complexity * complexity) / denom
            if denom > 0
            else 0.5
        )
        return 1.0 - combined

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self._max_concurrent)
        ]
        logger.info("QueuedToolScheduler started (%d workers)", self._max_concurrent)

    async def close(self) -> None:
        if not self._started:
            return
        for w in self._workers:
            w.cancel()
        self._workers = []
        queue = self._get_queue()
        while not queue.empty():
            entry = queue.get_nowait()
            if not entry.future.done():
                entry.future.cancel()
            queue.task_done()
        self._started = False
        logger.info("QueuedToolScheduler stopped")

    async def _worker(self, idx: int) -> None:
        queue = self._get_queue()
        while True:
            entry = await queue.get()
            logger.debug("Worker %d running score=%.3f", idx, entry.score)
            try:
                result = await entry.func()
                if not entry.future.done():
                    entry.future.set_result(result)
            except asyncio.CancelledError:
                if not entry.future.done():
                    entry.future.cancel()
                queue.task_done()
                raise
            except Exception:
                logger.exception("Tool execution failed in worker %d", idx)
                if not entry.future.done():
                    entry.future.set_exception(
                        asyncio.InvalidStateError("Tool execution failed; see logs")
                    )
            finally:
                queue.task_done()

    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        await self.start()

        queue = self._get_queue()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._counter += 1
        entry = _PendingCall(
            score=self._score(tool_name, chat_priority),
            counter=self._counter,
            entry_id=str(uuid.uuid4()),
            func=func,
            future=future,
        )
        await queue.put(entry)

        logger.debug(
            "Enqueued '%s' priority=%s score=%.3f",
            tool_name,
            chat_priority,
            entry.score,
        )

        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
