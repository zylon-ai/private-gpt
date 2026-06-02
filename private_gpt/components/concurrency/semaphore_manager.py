"""Semaphore manager abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class QueueShutdownError(Exception):
    pass


class SemaphoreManager(ABC):
    """Abstract semaphore manager interface."""

    @abstractmethod
    async def __aenter__(self) -> SemaphoreManager:
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, *_: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def start_processor(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        task_func: Callable[..., Awaitable[Any]],
        priority: int = 0,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
