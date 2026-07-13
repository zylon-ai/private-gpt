"""Chat schedulers — execution lifecycle management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from injector import Injector, inject, singleton

from private_gpt.arq.enqueue import abort_chat_job
from private_gpt.settings.settings import Settings

ChatSchedulerProvider = (
    type["BaseChatScheduler"] | Callable[[Injector], "BaseChatScheduler"]
)
_CHAT_SCHEDULERS: dict[str, ChatSchedulerProvider] = {}


def register_chat_scheduler(mode: str, provider: ChatSchedulerProvider) -> None:
    _CHAT_SCHEDULERS[mode] = provider


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


register_chat_scheduler("local", LocalChatScheduler)
register_chat_scheduler("arq", ArqChatScheduler)


@singleton
class ChatSchedulerFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector
        self._scheduler: BaseChatScheduler | None = None

        mode = self._settings.scheduler.chat.mode
        if mode not in _CHAT_SCHEDULERS:
            raise ValueError(f"Unknown scheduler.chat.mode: {mode}")

    def get(self) -> BaseChatScheduler:
        if self._scheduler is None:
            mode = self._settings.scheduler.chat.mode
            provider = _CHAT_SCHEDULERS[mode]
            self._scheduler = (
                self._injector.get(provider)
                if isinstance(provider, type)
                else provider(self._injector)
            )
        return self._scheduler
