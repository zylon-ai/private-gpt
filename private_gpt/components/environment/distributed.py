"""Distributed coordination for sandbox environments.

Provides a per-session mutex (serialising acquire/create/kill across
processes) and a shared last-activity clock (so reapers only kill sandboxes
idle everywhere). Uses ``redis_semaphore_async`` with an in-process fallback
when Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# In-memory fallback state, shared process-wide so every coordinator instance
# in this process coordinates against the same locks/activity. This mirrors
# Redis within a single process (used when Redis is unavailable/not configured).
_fallback_locks: dict[str, str] = {}
_fallback_activity: dict[str, float] = {}
_fallback_guard = asyncio.Lock()

# How long the per-session lock may be held (refreshed while held — a crashed
# holder therefore never blocks waiters for longer than this).
_LOCK_TTL_SECONDS = 120
# How long acquire() waits for another process to release the session lock.
_LOCK_WAIT_SECONDS = 300
# Shared last-activity keys never need to expire mid-session; 24h is ample.
_ACTIVITY_TTL_SECONDS = 60 * 60 * 24

# Namespace matching RedisSemaphoreManager's usage of redis_semaphore_async.
_SEM_NAMESPACE = "sandbox_lock"


class DistributedCoordinator:
    """Redis semaphore-based cross-process coordination with a memory fallback.

    All methods are safe to call when Redis is down: the in-memory fallback
    keeps single-process deployments fully functional, and Redis failures
    are logged once instead of raised.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._instance_id = instance_id or uuid.uuid4().hex[:8]
        self._redis: Redis | None = None
        self._redis_attempted = False
        self._warned = False
        self._semaphores: dict[str, object] = {}

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def enabled(self) -> bool:
        """True when connected to Redis (real cross-process coordination)."""
        return self._redis is not None

    async def _client(self) -> Redis | None:
        if self._redis is not None:
            return self._redis
        if self._redis_attempted:
            return None
        self._redis_attempted = True
        if not self._redis_url:
            return None
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(self._redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            logger.info(
                "Distributed coordinator connected to Redis at %s", self._redis_url
            )
        except Exception as exc:
            if not self._warned:
                logger.warning(
                    "Redis unavailable for sandbox coordination (%s); "
                    "falling back to in-process coordination",
                    exc,
                )
                self._warned = True
            self._redis = None
        return self._redis

    # ------------------------------------------------------------------
    # Per-session lock
    # ------------------------------------------------------------------

    def _semaphore(self, session_id: str) -> object:
        """Redis semaphore mutex for the session (redis_semaphore_async)."""
        from redis_semaphore_async import Semaphore  # type: ignore[import-untyped]

        sem = self._semaphores.get(session_id)
        if sem is None:
            sem = Semaphore(
                redis=self._redis,
                task_name=f"sandbox:{session_id}",
                value=1,
                namespace=_SEM_NAMESPACE,
            )
            self._semaphores[session_id] = sem
        return sem

    def _sem_counter_key(self, session_id: str) -> str:
        # Same key layout the library derives: {namespace}:{task_name}.
        return f"{_SEM_NAMESPACE}:sandbox:{session_id}"

    async def _refresh_lease(self, session_id: str) -> None:
        """Crash backstop: expire the semaphore counter while we hold it.

        If this process dies mid-hold, waiters recover after the TTL instead
        of waiting forever on a counter stuck at 0. Refreshed by a heartbeat
        while the lock is held.
        """
        redis = self._redis
        if redis is None:
            return
        with suppress(Exception):
            await redis.expire(self._sem_counter_key(session_id), _LOCK_TTL_SECONDS)

    async def _acquire_fallback(
        self, session_id: str, owner: str, deadline: float
    ) -> bool:
        key = f"sandbox:lock:{session_id}"
        while True:
            async with _fallback_guard:
                if _fallback_locks.get(key) is None:
                    _fallback_locks[key] = owner
                    return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    async def _release_fallback(self, session_id: str, owner: str) -> None:
        key = f"sandbox:lock:{session_id}"
        async with _fallback_guard:
            if _fallback_locks.get(key) == owner:
                _fallback_locks.pop(key, None)

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[bool]:
        """Wait for and hold the per-session lock for the duration of the block.

        Yields True when the cross-process lock is held. Yields False when the
        lock could not be acquired within ``_LOCK_WAIT_SECONDS`` (degraded
        mode) — callers should log and proceed with local coordination only.
        """
        owner = self._instance_id
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS

        redis = await self._client()
        if redis is None:
            acquired = await self._acquire_fallback(session_id, owner, deadline)
            try:
                yield acquired
            finally:
                if acquired:
                    await self._release_fallback(session_id, owner)
            return

        sem = self._semaphore(session_id)
        acquired = False
        heartbeat: asyncio.Task[None] | None = None
        try:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=_LOCK_WAIT_SECONDS)
                acquired = True
            except Exception:
                logger.warning(
                    "Could not acquire distributed lock for session %s within "
                    "%ds; proceeding without cross-process protection",
                    session_id,
                    _LOCK_WAIT_SECONDS,
                )
                acquired = False

            if acquired:
                await self._refresh_lease(session_id)

                async def _heartbeat() -> None:
                    while True:
                        await asyncio.sleep(_LOCK_TTL_SECONDS / 3)
                        await self._refresh_lease(session_id)

                heartbeat = asyncio.create_task(_heartbeat())

            yield acquired
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            if acquired:
                with suppress(Exception):
                    await sem.release()

    # ------------------------------------------------------------------
    # Shared last-activity clock
    # ------------------------------------------------------------------

    async def set_activity(self, session_id: str) -> None:
        """Record that this session was used right now (wall clock)."""
        redis = await self._client()
        key = f"sandbox:activity:{session_id}"
        if redis is None:
            async with _fallback_guard:
                _fallback_activity[key] = time.time()
            return
        with suppress(Exception):
            await redis.set(key, str(time.time()), ex=_ACTIVITY_TTL_SECONDS)

    async def get_activity(self, session_id: str) -> float | None:
        """Return the last recorded wall-clock usage, or None if unknown."""
        redis = await self._client()
        key = f"sandbox:activity:{session_id}"
        if redis is None:
            async with _fallback_guard:
                ts = _fallback_activity.get(key)
            return ts
        with suppress(Exception):
            raw = await redis.get(key)
            if raw is not None:
                return float(raw)
        return None

    async def close(self) -> None:
        self._semaphores.clear()
        if self._redis is not None:
            with suppress(Exception):
                await self._redis.aclose()
            self._redis = None
