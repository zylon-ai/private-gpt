from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from arq.constants import health_check_key_suffix
from arq.utils import timestamp_ms
from arq.worker import Worker
from redis.exceptions import ResponseError, WatchError

from private_gpt.arq.leases import (
    decode_in_progress_owner,
    encode_in_progress_owner,
    env_seconds,
    in_progress_key,
    worker_lease_key,
    worker_lease_seconds,
)

logger = logging.getLogger(__name__)

READINESS_FILE = Path("/tmp/arq_ready")
DEFAULT_ARQ_HEALTH_CHECK_INTERVAL_SECONDS = 5
STALE_LOCK_RECOVERY_INTERVAL_SECONDS = 2
_RELEASE_STALE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1]
   and redis.call('exists', KEYS[2]) == 0 then
    return redis.call('del', KEYS[1])
end
return 0
"""
_RELEASE_LEGACY_STALE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1]
   and redis.call('exists', KEYS[2]) == 0 then
    return redis.call('del', KEYS[1])
end
return 0
"""


def arq_health_check_interval() -> int:
    return env_seconds(
        "PGPT_ARQ_HEALTH_CHECK_INTERVAL_SECONDS",
        DEFAULT_ARQ_HEALTH_CHECK_INTERVAL_SECONDS,
    )


def arq_health_check_key(queue_name: str) -> str:
    worker_name = os.environ.get("HOSTNAME", "worker").strip() or "worker"
    return f"{queue_name}:health-check:{worker_name}"


def legacy_arq_health_check_key(queue_name: str) -> str:
    """Health key used by workers before per-pod native health keys."""
    return f"{queue_name}{health_check_key_suffix}"


def record_worker_ready() -> None:
    READINESS_FILE.touch()


def worker_is_ready() -> bool:
    return READINESS_FILE.is_file()


def clear_worker_liveness() -> None:
    READINESS_FILE.unlink(missing_ok=True)


class HeartbeatWorker(Worker):
    """ARQ worker with native health recording and reclaimable job ownership."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.worker_id = uuid.uuid4().hex
        self._lease_seconds = worker_lease_seconds()
        self._last_stale_lock_recovery = 0.0

    async def _renew_worker_lease(self) -> None:
        await self.pool.psetex(
            worker_lease_key(self.worker_id),
            self._lease_seconds * 1000,
            b"1",
        )

    async def start_jobs(self, job_ids: list[bytes]) -> None:
        """Claim jobs with this worker's identity instead of ARQ's legacy ``b"1"``.

        This mirrors ARQ 0.26's ``Worker.start_jobs`` (the supported version on
        this branch), but makes ownership atomic with the in-progress lock. A
        process killed between the lock transaction and ``run_job`` therefore
        still leaves a reclaimable lock.
        """
        for job_id_bytes in job_ids:
            await self.sem.acquire()

            if self.job_counter >= self.max_jobs:
                self.sem.release()
                return

            self.job_counter += 1
            job_id = job_id_bytes.decode()
            progress_key = in_progress_key(job_id)
            async with self.pool.pipeline(transaction=True) as pipe:
                await pipe.watch(progress_key)
                ongoing_exists = await pipe.exists(progress_key)
                score = await pipe.zscore(self.queue_name, job_id)
                if ongoing_exists or not score or score > timestamp_ms():
                    self.job_counter -= 1
                    self.sem.release()
                    logger.debug("job %s already running elsewhere", job_id)
                    continue

                pipe.multi()
                pipe.psetex(
                    progress_key,
                    int(self.in_progress_timeout_s * 1000),
                    encode_in_progress_owner(self.worker_id),
                )
                try:
                    await pipe.execute()
                except (ResponseError, WatchError):
                    self.job_counter -= 1
                    self.sem.release()
                    logger.debug(
                        "multi-exec error, job %s already started elsewhere", job_id
                    )
                else:
                    task = self.loop.create_task(self.run_job(job_id, int(score)))
                    task.add_done_callback(
                        lambda _: self._release_sem_dec_counter_on_complete()
                    )
                    self.tasks[job_id] = task

    async def _recover_stale_in_progress_jobs(self) -> int:
        now = time.monotonic()
        if now - self._last_stale_lock_recovery < STALE_LOCK_RECOVERY_INTERVAL_SECONDS:
            return 0
        self._last_stale_lock_recovery = now

        job_ids = await self.pool.zrangebyscore(
            self.queue_name,
            min=float("-inf"),
            max=timestamp_ms(),
            start=0,
            num=self.queue_read_limit,
        )
        if not job_ids:
            return 0

        decoded_ids = [
            job_id.decode() if isinstance(job_id, bytes) else str(job_id)
            for job_id in job_ids
        ]
        async with self.pool.pipeline(transaction=False) as pipe:
            for job_id in decoded_ids:
                pipe.get(in_progress_key(job_id))
            lock_values = await pipe.execute()

        owned_jobs: list[tuple[str, str, bytes | str]] = []
        legacy_jobs: list[tuple[str, bytes | str]] = []
        for job_id, lock_value in zip(decoded_ids, lock_values, strict=True):
            if lock_value is None:
                continue
            owner = decode_in_progress_owner(lock_value)
            if owner is None:
                legacy_jobs.append((job_id, lock_value))
            elif owner != self.worker_id:
                owned_jobs.append((job_id, owner, lock_value))

        lease_exists: list[object] = []
        if owned_jobs:
            async with self.pool.pipeline(transaction=False) as pipe:
                for _, owner, _ in owned_jobs:
                    pipe.exists(worker_lease_key(owner))
                lease_exists = await pipe.execute()

        recovered = 0
        for (job_id, owner, lock_value), owner_alive in zip(
            owned_jobs, lease_exists, strict=True
        ):
            if owner_alive:
                continue
            released = await self.pool.eval(
                _RELEASE_STALE_LOCK,
                2,
                in_progress_key(job_id),
                worker_lease_key(owner),
                lock_value,
            )
            if released:
                recovered += 1
                logger.warning(
                    "Recovered stale ARQ in-progress lock job_id=%s owner=%s",
                    job_id,
                    owner,
                )

        # Pre-upgrade ARQ workers stored b"1" rather than an owner id. Their
        # shared native health key proves that at least one legacy worker was
        # recently polling. Only reclaim unknown locks after that key expires.
        legacy_health_key = legacy_arq_health_check_key(self.queue_name)
        for job_id, lock_value in legacy_jobs:
            released = await self.pool.eval(
                _RELEASE_LEGACY_STALE_LOCK,
                2,
                in_progress_key(job_id),
                legacy_health_key,
                lock_value,
            )
            if released:
                recovered += 1
                logger.warning(
                    "Recovered legacy stale ARQ in-progress lock job_id=%s",
                    job_id,
                )
        return recovered

    async def _poll_iteration(self) -> None:
        # Renew before looking for stale owners. After a Redis interruption this
        # prevents this worker from reclaiming its own still-running jobs.
        await self._renew_worker_lease()
        await self._recover_stale_in_progress_jobs()
        await super()._poll_iteration()

    async def heart_beat(self) -> None:
        await self._renew_worker_lease()
        await super().heart_beat()
        # Worker.heart_beat() records ARQ's expiring Redis health key before
        # readiness is exposed to the HTTP sidecar.
        record_worker_ready()
