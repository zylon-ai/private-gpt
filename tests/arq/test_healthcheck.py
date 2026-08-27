import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from private_gpt.arq import healthcheck, liveness
from private_gpt.arq.healthcheck import (
    check_health,
    count_stale_pickable_jobs,
    health,
)
from private_gpt.arq.liveness import record_worker_heartbeat


@pytest.fixture
def liveness_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    heartbeat = tmp_path / "arq_worker_heartbeat"
    ready = tmp_path / "arq_ready"
    monkeypatch.setattr(liveness, "HEARTBEAT_FILE", heartbeat)
    monkeypatch.setattr(liveness, "READINESS_FILE", ready)
    return heartbeat, ready


def _write_heartbeat(
    files: tuple[Path, Path],
    *,
    ongoing: int,
    max_jobs: int = 2,
    age_seconds: float = 0,
) -> None:
    heartbeat, ready = files
    payload = {
        "ongoing": ongoing,
        "max_jobs": max_jobs,
        "ts": time.time() - age_seconds,
    }
    heartbeat.write_text(json.dumps(payload), encoding="utf-8")
    ready.touch()


async def test_healthcheck_is_unhealthy_before_worker_is_ready(
    liveness_files: tuple[Path, Path],
) -> None:
    del liveness_files
    status = await check_health()
    assert status["status"] == "unhealthy"
    assert status["reason"] == "not_ready"


async def test_healthcheck_is_unhealthy_when_idle_heartbeat_is_stale(
    liveness_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_ARQ_HEALTH_HEARTBEAT_STALE_SECONDS", "5")
    _write_heartbeat(liveness_files, ongoing=0, age_seconds=20)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "heartbeat_stale"


async def test_healthcheck_stays_healthy_when_busy_even_if_heartbeat_is_stale(
    liveness_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_ARQ_HEALTH_HEARTBEAT_STALE_SECONDS", "5")
    _write_heartbeat(liveness_files, ongoing=1, age_seconds=20)

    status = await check_health()

    assert status == {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
        "ongoing": 1,
        "max_jobs": 2,
    }


async def test_healthcheck_is_healthy_when_idle_and_queue_is_empty(
    liveness_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_heartbeat(liveness_files, ongoing=0)
    monkeypatch.setattr(
        healthcheck,
        "probe_stale_pickable_jobs",
        AsyncMock(return_value=(0, None)),
    )

    status = await check_health()

    assert status["status"] == "healthy"
    assert status["services"] == {"worker": "healthy"}


async def test_healthcheck_is_unhealthy_when_idle_with_pending_jobs(
    liveness_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_heartbeat(liveness_files, ongoing=0)
    monkeypatch.setattr(
        healthcheck,
        "probe_stale_pickable_jobs",
        AsyncMock(return_value=(3, None)),
    )

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "idle_with_pending_jobs"
    assert status["pending_jobs"] == 3


async def test_healthcheck_is_unhealthy_when_queue_probe_fails(
    liveness_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_heartbeat(liveness_files, ongoing=0)
    monkeypatch.setattr(
        healthcheck,
        "probe_stale_pickable_jobs",
        AsyncMock(return_value=(None, "redis_unreachable")),
    )

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "redis_unreachable"


async def test_health_endpoint_returns_503_when_unhealthy(
    liveness_files: tuple[Path, Path],
) -> None:
    del liveness_files
    with pytest.raises(HTTPException) as exc_info:
        await health()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["reason"] == "not_ready"


def test_record_worker_heartbeat_writes_ready_and_status(
    liveness_files: tuple[Path, Path],
) -> None:
    record_worker_heartbeat(ongoing=2, max_jobs=4)

    heartbeat, ready = liveness_files
    assert ready.is_file()
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["ongoing"] == 2
    assert payload["max_jobs"] == 4
    assert payload["ts"] <= time.time()


class _FakePipeline:
    def __init__(self, redis: "_FakeRedis") -> None:
        self._redis = redis
        self._keys: list[str] = []

    def exists(self, key: str) -> None:
        self._keys.append(key)

    async def execute(self) -> list[int]:
        return [1 if key in self._redis.in_progress else 0 for key in self._keys]

    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *args: object) -> None:
        del args


class _FakeRedis:
    def __init__(self, job_ids: list[bytes], in_progress: set[str]) -> None:
        self.job_ids = job_ids
        self.in_progress = in_progress

    async def zrangebyscore(self, *args: object, **kwargs: object) -> list[bytes]:
        del args, kwargs
        return self.job_ids

    def pipeline(self, transaction: bool = False) -> _FakePipeline:
        del transaction
        return _FakePipeline(self)


async def test_count_stale_pickable_jobs_ignores_in_progress_work() -> None:
    redis = _FakeRedis(
        job_ids=[b"job-a", b"job-b", b"job-c"],
        in_progress={"arq:in-progress:job-a", "arq:in-progress:job-c"},
    )

    assert (
        await count_stale_pickable_jobs(
            redis,
            queue_name="private_gpt:arq:queue:chat",
            cutoff_ms=1,
        )
        == 1
    )


async def test_count_stale_pickable_jobs_is_zero_when_queue_is_empty() -> None:
    redis = _FakeRedis(job_ids=[], in_progress=set())

    assert (
        await count_stale_pickable_jobs(
            redis,
            queue_name="private_gpt:arq:queue:chat",
            cutoff_ms=1,
        )
        == 0
    )
