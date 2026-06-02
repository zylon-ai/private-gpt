from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import settings

celery_settings = settings().celery


class BrokerConfig(BaseModel):
    url: str
    transport_options: dict[str, Any] = Field(default_factory=dict)


BrokerConfigProvider = Callable[[], BrokerConfig]


def _local_broker() -> BrokerConfig:
    return BrokerConfig(
        url="filesystem://",
        transport_options={
            "data_folder_in": local_data_path,
            "data_folder_out": local_data_path,
        },
    )


def _redis_broker() -> BrokerConfig:
    return BrokerConfig(
        url=f"{settings().redis.url}/2",
        transport_options={
            "max_connections": None,
            "socket_timeout": 30,
            "socket_connect_timeout": 30,
            "socket_keepalive": True,
        },
    )


def _rabbitmq_broker() -> BrokerConfig:
    return BrokerConfig(url=settings().rabbitmq.url)


_PROVIDERS: dict[str, BrokerConfigProvider] = {
    "local": _local_broker,
    "redis": _redis_broker,
    "rabbitmq": _rabbitmq_broker,
}


def register_broker_config(mode: str, provider: BrokerConfigProvider) -> None:
    _PROVIDERS[mode] = provider


def get_broker_config() -> BrokerConfig:
    provider = _PROVIDERS.get(celery_settings.broker_mode)
    if provider is None:
        raise ValueError(f"Invalid broker mode: {celery_settings.broker_mode}")
    return provider()
