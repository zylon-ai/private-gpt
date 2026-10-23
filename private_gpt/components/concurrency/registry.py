from __future__ import annotations

import logging
from collections.abc import Callable

from private_gpt.components.concurrency.semaphore_manager import SemaphoreManager
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

logger = logging.getLogger(__name__)

SemaphoreManagerProvider = Callable[
    [Settings | None, int | None, str | None], SemaphoreManager
]


def _redis_semaphore_provider(
    settings: Settings | None,
    max_concurrency: int | None,
    queue_key: str | None,
) -> SemaphoreManager:
    try:
        from private_gpt.components.concurrency.redis_semaphore_manager import (
            RedisSemaphoreManager,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message("Redis semaphore", extras="redis")
        ) from e

    logger.info("Using RedisSemaphoreManager for concurrency control")
    return RedisSemaphoreManager(
        settings=settings,
        max_concurrency=max_concurrency,
        queue_key=queue_key,
    )


def _memory_semaphore_provider(
    settings: Settings | None,
    max_concurrency: int | None,
    queue_key: str | None,
) -> SemaphoreManager:
    del settings
    from private_gpt.components.concurrency.memory_semaphore_manager import (
        MemorySemaphoreManager,
    )

    logger.info("Using MemorySemaphoreManager for concurrency control")
    return MemorySemaphoreManager(
        max_concurrency=max_concurrency,
        queue_key=queue_key,
    )


_PROVIDERS: dict[str, SemaphoreManagerProvider] = {
    "redis": _redis_semaphore_provider,
    "memory": _memory_semaphore_provider,
}


def register_semaphore_manager(name: str, provider: SemaphoreManagerProvider) -> None:
    _PROVIDERS[name] = provider


def create_semaphore_manager(
    settings: Settings | None = None,
    max_concurrency: int | None = None,
    queue_key: str | None = None,
) -> SemaphoreManager:
    settings = settings or get_global_injector().get(Settings)
    mode = settings.semaphore.mode
    provider = _PROVIDERS.get(mode)
    if provider is None:
        raise ValueError(
            f"Unsupported semaphore mode: {mode!r}. "
            f"Available: {', '.join(sorted(_PROVIDERS))}"
        )
    return provider(settings, max_concurrency, queue_key)
