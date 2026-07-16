import asyncio
from typing import Any

from private_gpt.components.web.scraper.pool import ScrapeSessionPool


class _FakeSession:
    pass


class _FakeSandboxProvider:
    def __init__(self) -> None:
        self.created: list[_FakeSession] = []
        self.killed: list[_FakeSession] = []

    async def create_session(self, **_: Any) -> _FakeSession:
        session = _FakeSession()
        self.created.append(session)
        return session

    async def kill_session(self, session: _FakeSession) -> None:
        self.killed.append(session)


def _pool(
    provider: _FakeSandboxProvider,
    *,
    pool_size: int = 2,
    max_requests_per_session: int = 1,
    idle_timeout_seconds: int = 300,
) -> ScrapeSessionPool:
    return ScrapeSessionPool(
        provider,  # type: ignore[arg-type]
        pool_size=pool_size,
        max_requests_per_session=max_requests_per_session,
        idle_timeout_seconds=idle_timeout_seconds,
    )


async def test_single_request_sessions_are_killed_on_release() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, max_requests_per_session=1)

    session = await pool.acquire()
    await pool.release(session)

    assert provider.killed == [session]

    other = await pool.acquire()
    assert other is not session, "exhausted sessions must not be reused"
    await pool.release(other)


async def test_sessions_are_reused_until_request_budget_is_spent() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, max_requests_per_session=3)

    first = await pool.acquire()
    await pool.release(first)
    assert provider.killed == []

    assert await pool.acquire() is first
    await pool.release(first)
    assert await pool.acquire() is first
    await pool.release(first)

    assert provider.killed == [first], "third release exhausts the budget"
    await pool.close()


async def test_batch_release_counts_all_requests() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, max_requests_per_session=3)

    session = await pool.acquire()
    await pool.release(session, requests=3)

    assert provider.killed == [session], "a batch of 3 exhausts the budget at once"
    await pool.close()


async def test_broken_sessions_are_killed_even_with_budget_left() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, max_requests_per_session=10)

    session = await pool.acquire()
    await pool.release(session, broken=True)

    assert provider.killed == [session]
    await pool.close()


async def test_pool_size_caps_concurrent_sessions() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, pool_size=1)

    first = await pool.acquire()

    waiter = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    assert not waiter.done(), "second acquire must wait for the free slot"

    await pool.release(first)
    second = await asyncio.wait_for(waiter, timeout=1)
    await pool.release(second)


async def test_close_kills_idle_sessions() -> None:
    provider = _FakeSandboxProvider()
    pool = _pool(provider, max_requests_per_session=5)

    session = await pool.acquire()
    await pool.release(session)
    assert provider.killed == []

    await pool.close()
    assert provider.killed == [session]
