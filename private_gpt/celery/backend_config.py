from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from private_gpt.settings.settings import settings

celery_settings = settings().celery


class BackendConfig(BaseModel):
    url: str
    transport_options: dict[str, Any] = Field(default_factory=dict)


BackendConfigProvider = Callable[[], BackendConfig]


def _local_backend() -> BackendConfig:
    return BackendConfig(url="db+sqlite:///celery_backend.db")


def _redis_backend() -> BackendConfig:
    return BackendConfig(url=f"{settings().redis.url}/4")


def _rabbitmq_backend() -> BackendConfig:
    return BackendConfig(url=settings().rabbitmq.url)


_PROVIDERS: dict[str, BackendConfigProvider] = {
    "local": _local_backend,
    "redis": _redis_backend,
    "rabbitmq": _rabbitmq_backend,
}


def register_backend_config(mode: str, provider: BackendConfigProvider) -> None:
    _PROVIDERS[mode] = provider


def get_backend_config() -> BackendConfig:
    provider = _PROVIDERS.get(celery_settings.backend_mode)
    if provider is None:
        raise ValueError(f"Invalid backend mode: {celery_settings.backend_mode}")
    return provider()
