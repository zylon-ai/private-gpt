"""Tests for the DistributedCoordinator (memory fallback path)."""

from __future__ import annotations

import asyncio
import time

import private_gpt.components.environment.distributed as _mod
from private_gpt.components.environment.distributed import DistributedCoordinator


async def _lock_wait(seconds: float):
    """Context manager temporarily shortening the lock-wait deadline."""

    class _Ctx:
        def __init__(self, secs: float) -> None:
            self.secs = secs
            self.orig = _mod._LOCK_WAIT_SECONDS

        async def __aenter__(self):
            _mod._LOCK_WAIT_SECONDS = self.secs
            return self

        async def __aexit__(self, *exc):
            _mod._LOCK_WAIT_SECONDS = self.orig

    ctx = _Ctx(seconds)
    await ctx.__aenter__()
    return ctx


async def test_session_lock_serialises_other_owner() -> None:
    """While owner A holds the session lock, owner B cannot acquire it."""
    c1 = DistributedCoordinator(instance_id="a")
    c2 = DistributedCoordinator(instance_id="b")

    holder_released = asyncio.Event()

    async def holder() -> None:
        async with c1.session_lock("s1") as ok:
            assert ok is True
            await asyncio.sleep(0.2)
        holder_released.set()

    async def waiter() -> None:
        async with c2.session_lock("s1") as ok:
            assert ok is True

    t1 = asyncio.create_task(holder())
    await asyncio.sleep(0.05)

    # B must NOT be able to take the lock while A holds it.
    t2 = asyncio.create_task(waiter())
    try:
        await asyncio.wait_for(asyncio.shield(t2), timeout=0.1)
        raise AssertionError("waiter acquired the lock while holder held it")
    except asyncio.TimeoutError:
        pass

    await t1
    await t2  # after A releases, B acquires and completes

    assert holder_released.is_set()


async def test_session_lock_releases_after_block() -> None:
    c1 = DistributedCoordinator(instance_id="a")
    c2 = DistributedCoordinator(instance_id="b")

    async with c1.session_lock("s2") as ok:
        assert ok is True
        # Another owner cannot take it while we hold it (short deadline).
        ctx = await _lock_wait(0.1)
        try:
            async with c2.session_lock("s2") as ok2:
                assert ok2 is False
        finally:
            await ctx.__aexit__(None, None, None)

    # Lock is released on exit — someone else can take it immediately.
    async with c2.session_lock("s2") as ok:
        assert ok is True


async def test_session_lock_yields_false_when_never_released() -> None:
    """A lock that never becomes available degrades to False after the wait
    deadline instead of hanging the caller forever."""
    c1 = DistributedCoordinator(instance_id="a")
    c2 = DistributedCoordinator(instance_id="b")

    ctx = await _lock_wait(0.15)
    try:
        async with c1.session_lock("s3"):
            async with c2.session_lock("s3") as ok:
                assert ok is False
    finally:
        await ctx.__aexit__(None, None, None)


async def test_shared_activity_clock() -> None:
    c = DistributedCoordinator(instance_id="a")

    assert await c.get_activity("s4") is None
    await c.set_activity("s4")
    ts = await c.get_activity("s4")
    assert ts is not None
    assert abs(ts - time.time()) < 5


async def test_cleanup_between_tests() -> None:
    _mod._fallback_locks.clear()
    _mod._fallback_activity.clear()
    assert _mod._fallback_locks == {}
    assert _mod._fallback_activity == {}
