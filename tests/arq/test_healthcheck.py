from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from private_gpt.arq import healthcheck, liveness
from private_gpt.arq.healthcheck import (
    ArqHealthState,
    DueJobCounts,
    check_health,
    health,
    inspect_due_jobs,
    parse_arq_health,
)
from private_gpt.arq.leases import encode_in_progress_owner, in_progress_key


@pytest.fixture(autouse=True)
def worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPT_ARQ_QUEUE", "chat")
    monkeypatch.setenv("PGPT_STATEFUL_WORKER_TYPE", "chat")
    monkeypatch.setenv("PGPT_ARQ_MAX_JOBS", "16")
    monkeypatch.setattr(healthcheck, "load_settings", _settings)


def _settings() -> MagicMock:
    current = MagicMock()
    current.scheduler.chat.celery_queue = "chat"
    return current


def _arq_health(*, ongoing: int = 0, queued: int = 0) -> ArqHealthState:
    return ArqHealthState(
        complete=1,
        failed=0,
        retried=0,
        ongoing=ongoing,
        queued=queued,
    )


def _patch_worker_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ongoing: int = 0,
    queued: int = 0,
    due_jobs: DueJobCounts | None = None,
    error: str | None = None,
) -> None:
    monkeypatch.setattr(liveness, "READINESS_FILE", MagicMock(is_file=lambda: True))
    monkeypatch.setattr(healthcheck, "worker_is_ready", lambda: True)
    monkeypatch.setattr(
        healthcheck,
        "probe_worker_state",
        AsyncMock(
            return_value=(
                None if error else _arq_health(ongoing=ongoing, queued=queued),
                None if error else (due_jobs or DueJobCounts()),
                error,
            )
        ),
    )
    monkeypatch.setattr(
        healthcheck,
        "probe_publisher_route",
        AsyncMock(return_value=None),
    )


def test_parse_arq_native_health_value() -> None:
    value = b"Aug-31 13:03:13 j_complete=4 j_failed=1 j_retried=2 j_ongoing=3 queued=5"

    assert parse_arq_health(value) == ArqHealthState(
        complete=4,
        failed=1,
        retried=2,
        ongoing=3,
        queued=5,
    )


def test_parse_arq_native_health_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="invalid ARQ health"):
        parse_arq_health("not an arq health value")


async def test_healthcheck_is_unhealthy_before_worker_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(healthcheck, "worker_is_ready", lambda: False)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "not_ready"


async def test_healthcheck_uses_missing_native_arq_key_as_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_state(monkeypatch, error="arq_health_missing")

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "arq_health_missing"


async def test_healthcheck_is_healthy_from_native_arq_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_state(monkeypatch, ongoing=2, queued=2)

    status = await check_health()

    assert status == {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
        "ongoing": 2,
        "queued": 2,
        "max_jobs": 16,
    }


async def test_healthcheck_is_unhealthy_when_idle_with_pending_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_state(
        monkeypatch,
        due_jobs=DueJobCounts(pickable=1),
        queued=1,
    )

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "idle_with_pending_jobs"
    assert status["pending_jobs"] == 1


async def test_healthcheck_allows_backlog_while_worker_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_state(
        monkeypatch,
        ongoing=2,
        queued=3,
        due_jobs=DueJobCounts(pickable=1),
    )

    status = await check_health()

    assert status["status"] == "healthy"
    assert status["ongoing"] == 2


@pytest.mark.parametrize(
    ("counts", "reason"),
    [
        (DueJobCounts(stale_in_progress=1), "stale_in_progress_jobs"),
        (DueJobCounts(unowned_in_progress=1), "unowned_in_progress_jobs"),
    ],
)
async def test_healthcheck_rejects_orphaned_in_progress_jobs(
    monkeypatch: pytest.MonkeyPatch,
    counts: DueJobCounts,
    reason: str,
) -> None:
    _patch_worker_state(monkeypatch, due_jobs=counts, queued=1)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == reason


async def test_healthcheck_rejects_queue_configuration_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_ARQ_QUEUE", "wrong-chat")
    monkeypatch.setattr(healthcheck, "worker_is_ready", lambda: True)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "queue_config_mismatch"
    assert status["expected_queue"].endswith(":chat")
    assert status["actual_queue"].endswith(":wrong-chat")


async def test_healthcheck_rejects_publisher_route_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_worker_state(monkeypatch)
    monkeypatch.setattr(
        healthcheck,
        "probe_publisher_route",
        AsyncMock(return_value={"status": "unhealthy", "reason": "mismatch"}),
    )

    status = await check_health()

    assert status == {"status": "unhealthy", "reason": "mismatch"}


async def test_http_health_returns_503_for_unhealthy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        healthcheck,
        "check_health",
        AsyncMock(return_value={"status": "unhealthy", "reason": "test"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await health()

    assert exc_info.value.status_code == 503


class _FakePipeline:
    def __init__(self, redis: _FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, str]] = []

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def get(self, key: str) -> None:
        self._commands.append(("get", key))

    def exists(self, key: str) -> None:
        self._commands.append(("exists", key))

    async def execute(self) -> list[object]:
        result: list[object] = []
        for command, key in self._commands:
            if command == "get":
                result.append(self._redis.locks.get(key))
            else:
                result.append(1 if key in self._redis.leases else 0)
        return result


class _FakeRedis:
    def __init__(
        self,
        *,
        job_ids: list[bytes],
        locks: dict[str, bytes],
        leases: set[str],
    ) -> None:
        self.job_ids = job_ids
        self.locks = locks
        self.leases = leases

    async def zrangebyscore(self, *args: object, **kwargs: object) -> list[bytes]:
        del args, kwargs
        return self.job_ids

    def pipeline(self, transaction: bool = False) -> _FakePipeline:
        del transaction
        return _FakePipeline(self)


async def test_inspect_due_jobs_distinguishes_lock_states() -> None:
    owner_alive = "worker-alive"
    owner_dead = "worker-dead"
    redis = _FakeRedis(
        job_ids=[b"pickable", b"alive", b"dead", b"legacy"],
        locks={
            in_progress_key("alive"): encode_in_progress_owner(owner_alive),
            in_progress_key("dead"): encode_in_progress_owner(owner_dead),
            in_progress_key("legacy"): b"1",
        },
        leases={f"private_gpt:arq:worker-lease:{owner_alive}"},
    )

    counts = await inspect_due_jobs(
        redis,
        queue_name="private_gpt:arq:queue:chat",
        cutoff_ms=1,
    )

    assert counts == DueJobCounts(
        pickable=1,
        stale_in_progress=1,
        unowned_in_progress=1,
    )
