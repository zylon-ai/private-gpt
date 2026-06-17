import logging
import queue
import threading
from time import sleep
from typing import NamedTuple

from pika import BlockingConnection, URLParameters
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from private_gpt.utils.retry import retry

logger = logging.getLogger(__name__)


class PublishJob(NamedTuple):
    exchange: str
    routing_key: str
    body: bytes


class BlockingPublisher(threading.Thread):
    """Simple publisher that will keep the connection alive on its own thread."""

    exchange: str
    parameters: URLParameters

    _connection: BlockingConnection | None
    _channel: BlockingChannel | None

    def __init__(self, exchange: str, parameters: URLParameters):
        super().__init__()
        self.daemon = True
        self.is_running = True

        self.exchange = exchange
        self.parameters = parameters

        # Queue of messages to publish
        self._publish_queue: queue.Queue[PublishJob] = queue.Queue()

        # Connection only exists in the publisher thread
        # Pika is not thread-safe, so we must ensure that
        # all operations on the connection and channel are done in the same thread
        self._connection = None
        self._channel = None

    @retry(
        AMQPConnectionError,
        tries=-1,
        delay=10,
        jitter=(1, 3),
        logger=logger,
    )
    def _ensure_connection(self, first_connection: bool = False) -> None:
        if (
            first_connection
            or self._connection is None
            or self._connection.is_closed
            or self._channel is None
            or self._channel.is_closed
        ):
            self._connection = BlockingConnection(self.parameters)
            self._channel = self._connection.channel()
            self._channel.exchange_declare(
                self.exchange, durable=True, auto_delete=False, exchange_type="topic"
            )

    def _publish(self) -> bool:
        sent_messages = 0
        while not self._publish_queue.empty():
            job = self._publish_queue.get_nowait()

            # Ensure the connection is alive before publishing
            self._ensure_connection()
            assert self._channel is not None

            self._channel.basic_publish(job.exchange, job.routing_key, body=job.body)
            self._publish_queue.task_done()
            sent_messages += 1

        return bool(sent_messages)

    def run(self) -> None:
        self._ensure_connection(first_connection=True)
        while self.is_running:
            try:
                sent_messages = self._publish()

                # Keep connection alive even if no messages were sent
                if not sent_messages:
                    sleep(5)

                    self._ensure_connection()
                    assert self._connection is not None
                    self._connection.process_data_events(time_limit=1)

            except Exception as e:
                # This will happen if connection to rabbitmq is lost
                # Must keep the thread alive
                logger.error(f"Error in publisher thread: {e}")

        # Clean up the resources after stopping the thread
        try:
            if self._connection is not None and self._connection.is_open:
                self._publish()
                self._connection.process_data_events(time_limit=1)
                self._connection.close()
        except Exception as e:
            logger.error(f"Error while closing connection: {e}")
            self._publish_queue.queue.clear()

    def publish(self, exchange: str, routing_key: str, body: bytes) -> None:
        if self.is_running:
            job = PublishJob(exchange, routing_key, body)
            self._publish_queue.put(job)

    def drain(self) -> None:
        """Wait for all queued messages to be published without stopping the thread."""
        self._publish_queue.join()

    def join(self, timeout: float | None = None) -> None:
        """Stop the publisher thread and wait for it to finish."""
        self.is_running = False
        self._publish_queue.join()
        super().join(timeout)

    def close(self) -> None:
        """Close the connection and stop the thread."""
        logger.debug("Closing BlockingPublisher")
        self.join(timeout=30.0)
        logger.debug("BlockingPublisher closed")
