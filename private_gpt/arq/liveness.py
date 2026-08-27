from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arq.worker import Worker

READINESS_FILE = Path("/tmp/arq_ready")
HEARTBEAT_FILE = Path("/tmp/arq_worker_heartbeat")


@dataclass(frozen=True)
class WorkerHeartbeat:
    ongoing: int
    max_jobs: int
    ts: float


def record_worker_heartbeat(*, ongoing: int, max_jobs: int) -> None:
    payload = {
        "ongoing": ongoing,
        "max_jobs": max_jobs,
        "ts": time.time(),
    }
    HEARTBEAT_FILE.write_text(json.dumps(payload), encoding="utf-8")
    READINESS_FILE.touch()


def read_worker_heartbeat() -> WorkerHeartbeat | None:
    if not READINESS_FILE.is_file() or not HEARTBEAT_FILE.is_file():
        return None

    payload: dict[str, Any] = json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
    return WorkerHeartbeat(
        ongoing=int(payload["ongoing"]),
        max_jobs=int(payload.get("max_jobs", 0)),
        ts=float(payload["ts"]),
    )


def clear_worker_liveness() -> None:
    HEARTBEAT_FILE.unlink(missing_ok=True)
    READINESS_FILE.unlink(missing_ok=True)


class HeartbeatWorker(Worker):
    """ARQ worker that exposes poll-loop liveness for the sidecar healthcheck."""

    async def heart_beat(self) -> None:
        record_worker_heartbeat(
            ongoing=sum(not task.done() for task in self.tasks.values()),
            max_jobs=self.max_jobs,
        )
        await super().heart_beat()
