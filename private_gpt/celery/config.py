from typing import Any, ClassVar

from private_gpt.celery.backend_config import BackendConfig, get_backend_config
from private_gpt.celery.broker_config import BrokerConfig, get_broker_config
from private_gpt.settings.settings import settings

celery_settings = settings().celery
celery_settings.validate_config()

broker_config: BrokerConfig = get_broker_config()
backend_config: BackendConfig = get_backend_config()


class CeleryConfig:
    """Celery configuration settings.

    This class defines the configuration settings for the Celery application.
    Celery documentation:
    https://docs.celeryq.dev/en/stable/userguide/configuration.html
    Broker configuration took from:
    https://stackoverflow.com/questions/44716178/celery-could-not-connect-to-rabbitmq
    More useful information:
    https://engineering.instawork.com/celery-eta-tasks-demystified-424b836e4e94
    """

    # ===== Serialization Configuration =====
    # Define how tasks and results are serialized/deserialized
    task_serializer = "pickle"
    result_serializer = "pickle"
    event_serializer = "json"

    # Define accepted content types for messages
    accept_content: ClassVar[list[str]] = [
        "application/json",
        "application/x-python-serialize",
    ]
    result_accept_content: ClassVar[list[str]] = [
        "application/json",
        "application/x-python-serialize",
    ]

    # ===== Timeout Configuration =====
    time_limit = celery_settings.hard_time_limit  # Time limit for tasks
    soft_time_limit = celery_settings.soft_time_limit  # Soft time limit for tasks
    visibility_timeout = (
        celery_settings.visibility_timeout
    )  # Visibility timeout for tasks

    # ===== Task Configuration =====
    task_store_eager_result = True  # Store eager results in memory
    task_track_started = True  # Track task start events
    task_store_errors_even_if_ignored = True  # Store errors even if ignored

    # ===== Broker (Message Queue) Configuration =====
    # Heartbeat settings to maintain broker connection
    broker_heartbeat = 30  # Heartbeat value in seconds to detect connection issues
    broker_heartbeat_checkrate = 10.0  # How often to check the heartbeat

    # Connection pool settings
    broker_pool_limit: int | None = (
        None  # Disable connection pooling to prevent connection reset issues
    )

    # Connection retry settings
    broker_connection_timeout = 10  # Timeout for broker connection attempts (seconds)
    broker_connection_retry = True  # Enable connection retry on failure
    broker_connection_max_retries: int | None = None  # Retry forever (no limit)
    broker_connection_retry_on_startup = True  # Retry connection attempts on startup

    # Advanced broker transport options
    broker_transport_options: ClassVar[dict[str, Any]] = {
        "visibility_timeout": celery_settings.visibility_timeout
        if celery_settings.visibility_timeout
        else None,
        **broker_config.transport_options,
    }

    # ===== Result Backend Configuration =====
    result_backend_always_retry = True  # Retry results on failure
    result_backend_max_retries: int | None = None  # Retry forever (no limit)

    # Advanced result backend options
    result_backend_transport_options: ClassVar[dict[str, Any]] = {
        "visibility_timeout": celery_settings.visibility_timeout
        if celery_settings.visibility_timeout
        else None,
        **backend_config.transport_options,
    }

    # ===== Task Execution Configuration =====
    task_acks_late = (
        celery_settings.acks_late
    )  # Acknowledge tasks after execution (from settings)
    task_reject_on_worker_lost = True  # Reject tasks if worker connection is lost

    # ===== Worker Configuration =====
    # Prefetch settings
    worker_prefetch_multiplier = (
        1  # Fetch one task at a time to prevent worker overload
    )

    # Event settings
    worker_send_task_events = True  # Enable sending task events to consume from Flower
    worker_max_tasks_per_child = (
        1  # Each worker process handles one task to prevent memory leaks.
    )
