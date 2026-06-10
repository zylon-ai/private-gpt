import atexit
import logging
import threading
from collections.abc import Callable

import pika
from injector import inject, singleton

from private_gpt.components.broker.blocking_publisher import BlockingPublisher
from private_gpt.settings.settings import Settings, settings

logger = logging.getLogger(__name__)

BrokerProvider = Callable[[Settings], BlockingPublisher | None]


def _rabbitmq_broker_provider(settings: Settings) -> BlockingPublisher:
    publisher = BlockingPublisher(
        exchange=BrokerComponent.main_exchange,
        parameters=pika.URLParameters(settings.rabbitmq.url),
    )
    publisher.start()
    return publisher


_PROVIDERS: dict[str, BrokerProvider] = {
    "rabbitmq": _rabbitmq_broker_provider,
}


def register_broker(mode: str, provider: BrokerProvider) -> None:
    _PROVIDERS[mode] = provider


class BrokerInstanceRegistry:

    _settings: Settings
    _instances: dict[str, BlockingPublisher | None]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._instances = {}
        self._lock = threading.Lock()

    def get_or_create(self, mode: str) -> BlockingPublisher | None:
        instance = self._instances.get(mode)
        if instance is not None and instance.is_alive():
            return instance

        with self._lock:
            instance = self._instances.get(mode)
            if instance is not None and instance.is_alive():
                return instance

            provider = _PROVIDERS.get(mode)
            if provider is None:
                raise ValueError(f"Unsupported task result broker mode: {mode}")

            new_instance = provider(self._settings)
            self._instances[mode] = new_instance
            return new_instance

    def close(self) -> None:
        with self._lock:
            modes = list(self._instances.keys())
            for mode in modes:
                instance = self._instances.pop(mode, None)
                if instance is not None:
                    instance.close()


INSTANCE_REGISTRY = BrokerInstanceRegistry(settings())
atexit.register(INSTANCE_REGISTRY.close)


@singleton
class BrokerComponent:
    blocking_publisher: BlockingPublisher | None = None
    main_exchange: str = "main"

    @inject
    def __init__(self, settings: Settings) -> None:
        self.blocking_publisher = INSTANCE_REGISTRY.get_or_create(
            settings.tasks_results_broker.mode
        )

    def publish(self, exchange: str, routing_key: str, body: bytes) -> None:
        if self.blocking_publisher is not None:
            logger.debug(
                f"Sending message to ${exchange}/{routing_key}, {len(body)} bytes"
            )
            self.blocking_publisher.publish(exchange, routing_key, body)
            logger.info(f"Message sent to ${exchange}/{routing_key}, {len(body)} bytes")

    def join(self) -> None:
        """Drain all queued messages without stopping the publisher thread."""
        if self.blocking_publisher is not None:
            logger.debug("Draining blocking publisher")
            self.blocking_publisher.drain()

    def close(self) -> None:
        """No-op: publisher lifecycle is owned by INSTANCE_REGISTRY."""
        pass
