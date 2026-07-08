import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from celery import bootsteps  # type: ignore
from celery.signals import (
    worker_process_init,
    worker_process_shutdown,
    worker_ready,
    worker_shutdown,
)

logger = logging.getLogger(__name__)

# Files for health checks
READINESS_FILE = Path("/tmp/celery_ready")
HEARTBEAT_FILE = Path("/tmp/celery_worker_heartbeat")


class LivenessProbe(bootsteps.StartStopStep):  # type: ignore
    """Liveness probe for Celery worker.

    Code adapted from:
    https://github.com/celery/celery/issues/4079#issuecomment-1270085680
    """

    requires: ClassVar[set[str]] = {"celery.worker.components:Timer"}

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.tref = None

    def start(self, worker: Any) -> None:
        self.tref = worker.timer.call_repeatedly(
            1.0, self.update_heartbeat_file, (worker,), priority=10  # Every second
        )

    def stop(self, worker: Any) -> None:
        HEARTBEAT_FILE.unlink(missing_ok=True)

    def update_heartbeat_file(self, worker: Any) -> None:
        HEARTBEAT_FILE.touch()


def _warm_chat_worker() -> None:
    """Eagerly warm the full DI for a long-lived chat worker.

    Only runs when ``PGPT_CHAT_WORKER=true`` is set in the environment (the
    chat worker entrypoint sets it). The components are singletons, so they
    are resolved once here and reused for the entire worker lifetime
    (``--max-tasks-per-child=None``).
    """
    if os.getenv("PGPT_CHAT_WORKER", "false").lower() != "true":
        return

    from private_gpt.celery.base import ChatBackgroundTask

    logger.info("PGPT_CHAT_WORKER=true: eagerly warming chat worker runtime")
    ChatBackgroundTask.warm_up()
    logger.info("Chat worker DI warmed successfully")


@worker_ready.connect
def handle_worker_ready(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker ready event."""
    is_chat_worker = os.getenv("PGPT_CHAT_WORKER", "false").lower() == "true"
    if is_chat_worker and os.getenv("PGPT_CELERY_POOL", "prefork") != "solo":
        return

    if is_chat_worker:
        _warm_chat_worker()

    READINESS_FILE.touch()


@worker_process_init.connect
def handle_worker_process_init(**kwargs: dict[str, Any]) -> None:
    """Warm each prefork child that will execute chat tasks."""
    _warm_chat_worker()
    if os.getenv("PGPT_CHAT_WORKER", "false").lower() == "true":
        READINESS_FILE.touch()


@worker_process_shutdown.connect
def handle_worker_process_shutdown(**kwargs: dict[str, Any]) -> None:
    """Clean up chat worker loop resources on child shutdown."""
    if os.getenv("PGPT_CHAT_WORKER", "false").lower() != "true":
        return

    from private_gpt.celery.base import ChatBackgroundTask

    ChatBackgroundTask.shutdown_runtime()


@worker_shutdown.connect
def handle_worker_shutdown(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker shutdown event."""
    READINESS_FILE.unlink(missing_ok=True)
    HEARTBEAT_FILE.unlink(missing_ok=True)
