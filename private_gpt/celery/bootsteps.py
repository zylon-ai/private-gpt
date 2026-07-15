import fcntl
import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from celery import bootsteps  # ty:ignore[unresolved-import]
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

STATEFUL_WARMUP_LOCK = Path("/tmp/stateful_warmup.lock")


class LivenessProbe(bootsteps.StartStopStep):
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
            1.0,
            self.update_heartbeat_file,
            (worker,),
            priority=10,  # Every second
        )

    def stop(self, worker: Any) -> None:
        HEARTBEAT_FILE.unlink(missing_ok=True)

    def update_heartbeat_file(self, worker: Any) -> None:
        HEARTBEAT_FILE.touch()


def _is_stateful() -> bool:
    return bool(os.getenv("PGPT_STATEFUL_WORKER_TYPE", "").strip())


def _warm_stateful_worker() -> None:
    worker_type = os.getenv("PGPT_STATEFUL_WORKER_TYPE", "").strip()
    if not worker_type:
        return

    from private_gpt.celery.base import StatefulBackgroundTask

    logger.info(
        "STATEFUL_WORKER_TYPE=%s: eagerly warming StatefulBackgroundTask",
        worker_type,
    )

    STATEFUL_WARMUP_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(STATEFUL_WARMUP_LOCK), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        StatefulBackgroundTask.warm_up()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    logger.info("StatefulBackgroundTask DI warmed successfully")


def _shutdown_stateful_worker() -> None:
    worker_type = os.getenv("PGPT_STATEFUL_WORKER_TYPE", "").strip()
    if not worker_type:
        return

    from private_gpt.celery.base import StatefulBackgroundTask

    StatefulBackgroundTask.shutdown_runtime()


@worker_ready.connect
def handle_worker_ready(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker ready event."""
    if _is_stateful():
        _warm_stateful_worker()

    READINESS_FILE.touch()


@worker_process_init.connect
def handle_worker_process_init(**kwargs: dict[str, Any]) -> None:
    """Warm each prefork child that will execute stateful tasks.

    The warm-up is serialized via a file lock so children load models
    one at a time, avoiding OOM from simultaneous loading.
    """
    if _is_stateful():
        _warm_stateful_worker()


@worker_process_shutdown.connect
def handle_worker_process_shutdown(**kwargs: dict[str, Any]) -> None:
    """Clean up stateful worker loop resources on child shutdown."""
    if not _is_stateful():
        return

    _shutdown_stateful_worker()


@worker_shutdown.connect
def handle_worker_shutdown(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker shutdown event."""
    READINESS_FILE.unlink(missing_ok=True)
    HEARTBEAT_FILE.unlink(missing_ok=True)
