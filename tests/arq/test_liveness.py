from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from arq.worker import Worker

if TYPE_CHECKING:
    import pytest

from private_gpt.arq import liveness
from private_gpt.arq.leases import encode_in_progress_owner, in_progress_key
from private_gpt.arq.liveness import HeartbeatWorker


async def test_heartbeat_worker_uses_native_arq_health_and_marks_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "arq_ready"
    monkeypatch.setattr(liveness, "READINESS_FILE", ready)

    native_heartbeat = AsyncMock()
    monkeypatch.setattr(Worker, "heart_beat", native_heartbeat)

    async def noop(ctx: object) -> None:
        del ctx

    pool = MagicMock()
    pool.psetex = AsyncMock()
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
    )

    await worker.heart_beat()

    native_heartbeat.assert_awaited_once_with()
    pool.psetex.assert_awaited_once()
    assert ready.is_file()


def test_native_health_key_is_unique_per_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOSTNAME", "chat-worker-pod")

    assert liveness.arq_health_check_key("queue") == (
        "queue:health-check:chat-worker-pod"
    )


class _ClaimPipeline:
    def __init__(self) -> None:
        self.claim: tuple[str, int, bytes] | None = None

    async def __aenter__(self) -> _ClaimPipeline:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def watch(self, key: str) -> None:
        del key

    async def exists(self, key: str) -> int:
        del key
        return 0

    async def zscore(self, queue: str, job_id: str) -> int:
        del queue, job_id
        return 1

    def multi(self) -> None:
        pass

    def psetex(self, key: str, ttl_ms: int, value: bytes) -> None:
        self.claim = (key, ttl_ms, value)

    async def execute(self) -> list[object]:
        return []


async def test_worker_claims_arq_lock_with_owner_atomically() -> None:
    async def noop(ctx: object) -> None:
        del ctx

    pipeline = _ClaimPipeline()
    pool = MagicMock()
    pool.pipeline = MagicMock(return_value=pipeline)
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
    )
    worker.run_job = AsyncMock()  # type: ignore[method-assign]

    await worker.start_jobs([b"job-id"])
    await asyncio.gather(*worker.tasks.values())

    assert pipeline.claim == (
        in_progress_key("job-id"),
        int(worker.in_progress_timeout_s * 1000),
        encode_in_progress_owner(worker.worker_id),
    )
    worker.run_job.assert_awaited_once_with("job-id", 1)


class _Pipeline:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses

    async def __aenter__(self) -> _Pipeline:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def get(self, key: str) -> None:
        del key

    def exists(self, key: str) -> None:
        del key

    async def execute(self) -> list[object]:
        return self.responses


async def test_worker_recovers_lock_when_owner_lease_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def noop(ctx: object) -> None:
        del ctx

    dead_owner = "dead-worker"
    pool = MagicMock()
    pool.zrangebyscore = AsyncMock(return_value=[b"job-id"])
    pool.pipeline = MagicMock(
        side_effect=[
            _Pipeline([encode_in_progress_owner(dead_owner)]),
            _Pipeline([0]),
        ]
    )
    pool.eval = AsyncMock(return_value=1)
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
    )
    monkeypatch.setattr(worker, "_last_stale_lock_recovery", 0.0)

    recovered = await worker._recover_stale_in_progress_jobs()

    assert recovered == 1
    pool.eval.assert_awaited_once()


async def test_worker_does_not_recover_live_owner_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def noop(ctx: object) -> None:
        del ctx

    live_owner = "live-worker"
    pool = MagicMock()
    pool.zrangebyscore = AsyncMock(return_value=[b"job-id"])
    pool.pipeline = MagicMock(
        side_effect=[
            _Pipeline([encode_in_progress_owner(live_owner)]),
            _Pipeline([1]),
        ]
    )
    pool.eval = AsyncMock(return_value=1)
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
    )
    monkeypatch.setattr(worker, "_last_stale_lock_recovery", 0.0)

    recovered = await worker._recover_stale_in_progress_jobs()

    assert recovered == 0
    pool.eval.assert_not_awaited()


async def test_worker_recovers_legacy_lock_when_legacy_health_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def noop(ctx: object) -> None:
        del ctx

    pool = MagicMock()
    pool.zrangebyscore = AsyncMock(return_value=[b"job-id"])
    pool.pipeline = MagicMock(return_value=_Pipeline([b"1"]))
    pool.eval = AsyncMock(return_value=1)
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
        queue_name="chat-queue",
    )
    monkeypatch.setattr(worker, "_last_stale_lock_recovery", 0.0)

    recovered = await worker._recover_stale_in_progress_jobs()

    assert recovered == 1
    pool.eval.assert_awaited_once()
    assert pool.eval.await_args.args[3] == "chat-queue:health-check"


async def test_worker_keeps_legacy_lock_while_legacy_health_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def noop(ctx: object) -> None:
        del ctx

    pool = MagicMock()
    pool.zrangebyscore = AsyncMock(return_value=[b"job-id"])
    pool.pipeline = MagicMock(return_value=_Pipeline([b"1"]))
    pool.eval = AsyncMock(return_value=0)
    worker = HeartbeatWorker(
        functions=[noop],
        redis_pool=pool,
        handle_signals=False,
        queue_name="chat-queue",
    )
    monkeypatch.setattr(worker, "_last_stale_lock_recovery", 0.0)

    recovered = await worker._recover_stale_in_progress_jobs()

    assert recovered == 0
    pool.eval.assert_awaited_once()
