from pathlib import Path
from typing import Any, ClassVar

from celery import bootsteps  # type: ignore
from celery.signals import worker_ready, worker_shutdown

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


@worker_ready.connect
def handle_worker_ready(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker ready event."""
    READINESS_FILE.touch()


@worker_shutdown.connect
def handle_worker_shutdown(**kwargs: dict[str, Any]) -> None:
    """Signal handler for worker shutdown event."""
    READINESS_FILE.unlink(missing_ok=True)
    HEARTBEAT_FILE.unlink(missing_ok=True)
