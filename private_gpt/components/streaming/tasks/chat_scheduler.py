"""Chat schedulers — execution lifecycle management."""

from __future__ import annotations

from abc import ABC, abstractmethod

from injector import inject, singleton

from private_gpt.arq.enqueue import abort_chat_job
from private_gpt.settings.settings import Settings


class BaseChatScheduler(ABC):
    """Provides lifecycle operations for chat executions."""

    @abstractmethod
    async def cancel(self, correlation_id: str) -> bool:
        """Cancel a running execution."""


class LocalChatScheduler(BaseChatScheduler):
    """Cancels local asyncio tasks by name."""

    async def cancel(self, correlation_id: str) -> bool:
        import asyncio

        for task in asyncio.all_tasks():
            if task.get_name() == f"chat_{correlation_id}":
                task.cancel()
                return True
        return False


class ArqChatScheduler(BaseChatScheduler):
    """Aborts ARQ jobs."""

    async def cancel(self, correlation_id: str) -> bool:
        return await abort_chat_job(correlation_id=correlation_id)


@singleton
class ChatSchedulerFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        mode = settings.scheduler.chat.mode
        if mode == "local":
            self._scheduler: BaseChatScheduler = LocalChatScheduler()
        elif mode == "arq":
            self._scheduler = ArqChatScheduler()
        else:
            raise ValueError(f"Unknown scheduler.chat.mode: {mode}")

    def get(self) -> BaseChatScheduler:
        return self._scheduler
