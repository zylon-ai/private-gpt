import logging
from collections.abc import Callable

import pika
from injector import inject, singleton

from private_gpt.components.broker.blocking_publisher import BlockingPublisher
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

BrokerProvider = Callable[[Settings], BlockingPublisher | None]


def _rabbitmq_broker_provider(settings: Settings) -> BlockingPublisher:
    publisher = BlockingPublisher(
        exchange=BrokerComponent.main_exchange,
        parameters=pika.URLParameters(settings.rabbitmq.url),
    )
    publisher.start()
    return publisher


def _none_broker_provider(settings: Settings) -> None:
    del settings
    return None


_PROVIDERS: dict[str, BrokerProvider] = {
    "rabbitmq": _rabbitmq_broker_provider,
}


def register_broker(mode: str, provider: BrokerProvider) -> None:
    _PROVIDERS[mode] = provider


@singleton
class BrokerComponent:
    blocking_publisher: BlockingPublisher | None = None
    main_exchange: str = "main"

    @inject
    def __init__(self, settings: Settings) -> None:
        provider = _PROVIDERS.get(settings.tasks_results_broker.mode)
        if provider is None:
            raise ValueError(
                f"Unsupported task result broker mode: {settings.tasks_results_broker.mode}"
            )
        self.blocking_publisher = provider(settings) or None

    def publish(self, exchange: str, routing_key: str, body: bytes) -> None:
        # If there is no configured broker, we just ignore the publishing
        if self.blocking_publisher is not None:
            logger.debug(
                f"Sending message to ${exchange}/{routing_key}, {len(body)} bytes"
            )
            self.blocking_publisher.publish(exchange, routing_key, body)
            logger.info(f"Message sent to ${exchange}/{routing_key}, {len(body)} bytes")

    def join(self) -> None:
        """Join the blocking publisher thread if it exists."""
        if self.blocking_publisher is not None:
            logger.debug("Joining blocking publisher")
            self.blocking_publisher.join(timeout=30)

    def close(self) -> None:
        """Close the blocking publisher if it exists."""
        if self.blocking_publisher is not None:
            logger.debug("Closing blocking publisher")
            self.blocking_publisher.close()
            del self.blocking_publisher
