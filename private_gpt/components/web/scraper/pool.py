from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

from private_gpt.components.web.scraper.base import WebScraperProvider
from private_gpt.components.web.scraper.batcher import ScrapeBatcher
from private_gpt.components.web.scraper.runner import (
    build_scrape_config,
    run_scrape_in_session,
)

if TYPE_CHECKING:
    from private_gpt.components.sandbox.base import SandboxProvider, SandboxSession
    from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_REAPER_INTERVAL_SECONDS = 60


class _PooledSession:
    def __init__(self, session: SandboxSession) -> None:
        self.session = session
        self.request_count = 0
        self.last_used = time.monotonic()


class ScrapeSessionPool:
    """Bounded pool of sandbox sessions with request-count recycling.

    ``max_requests_per_session`` governs statefulness: ``1`` destroys the
    session after every scrape (stateless), ``N`` keeps it warm for N scrapes
    before recycling it. Idle warm sessions are reaped after
    ``idle_timeout_seconds``.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        *,
        pool_size: int,
        max_requests_per_session: int,
        idle_timeout_seconds: int,
        user_id: str = "web-scraper",
        session_timeout_seconds: int | None = None,
    ) -> None:
        self._provider = provider
        self._max_requests = max(1, max_requests_per_session)
        self._idle_timeout = idle_timeout_seconds
        self._user_id = user_id
        self._session_timeout = session_timeout_seconds
        self._semaphore = asyncio.Semaphore(max(1, pool_size))
        self._lock = asyncio.Lock()
        self._idle: list[_PooledSession] = []
        self._busy: dict[int, _PooledSession] = {}
        self._reaper: asyncio.Task[None] | None = None
        self._closed = False

    async def acquire(self) -> SandboxSession:
        await self._semaphore.acquire()
        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("Scrape session pool is closed")
                if self._idle:
                    pooled = self._idle.pop()
                    self._busy[id(pooled.session)] = pooled
                    return pooled.session

            session = await self._provider.create_session(
                user_id=self._user_id, timeout=self._session_timeout
            )
            async with self._lock:
                self._busy[id(session)] = _PooledSession(session)
            if self._max_requests > 1:
                self._ensure_reaper()
            return session
        except BaseException:
            self._semaphore.release()
            raise

    async def release(
        self, session: SandboxSession, *, broken: bool = False, requests: int = 1
    ) -> None:
        """Return a session; ``requests`` is how many scrapes this use served."""
        try:
            async with self._lock:
                pooled = self._busy.pop(id(session), None)
            if pooled is None:
                logger.warning("Released a session unknown to the pool, killing it")
                await self._kill(session)
                return
            pooled.request_count += max(1, requests)
            pooled.last_used = time.monotonic()
            if broken or self._closed or pooled.request_count >= self._max_requests:
                await self._kill(session)
                return
            async with self._lock:
                self._idle.append(pooled)
        finally:
            self._semaphore.release()

    async def _kill(self, session: SandboxSession) -> None:
        try:
            await self._provider.kill_session(session)
        except Exception:
            logger.warning("Failed to kill scrape session", exc_info=True)

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_idle())

    async def _reap_idle(self) -> None:
        while not self._closed:
            await asyncio.sleep(_REAPER_INTERVAL_SECONDS)
            cutoff = time.monotonic() - self._idle_timeout
            async with self._lock:
                expired = [p for p in self._idle if p.last_used <= cutoff]
                self._idle = [p for p in self._idle if p.last_used > cutoff]
            for pooled in expired:
                logger.debug("Reaping idle scrape session")
                await self._kill(pooled.session)

    async def close(self) -> None:
        self._closed = True
        if self._reaper is not None and not self._reaper.done():
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        self._reaper = None
        async with self._lock:
            idle, self._idle = self._idle, []
        for pooled in idle:
            await self._kill(pooled.session)


class PooledWebScraperProvider(WebScraperProvider):
    """WebScraperProvider that runs the scrape script in pooled sandbox sessions.

    Concrete providers supply the pool (which wraps a ``SandboxProvider``) and
    the writable directory inside the session where script/config/output live.
    Requests are coalesced by a :class:`ScrapeBatcher` so near-simultaneous
    scrapes share one session/browser run instead of launching one each.
    """

    def __init__(
        self, settings: Settings, pool: ScrapeSessionPool, base_dir: str
    ) -> None:
        super().__init__(settings)
        self._pool = pool
        self._base_dir = base_dir
        self._batcher = ScrapeBatcher(
            batch_size=settings.web_fetch.batch_size,
            batch_wait_seconds=settings.web_fetch.batch_wait_ms / 1000,
            run_batch=self._run_batch,
        )

    async def scrape_html(self, url: str, timeout_seconds: int) -> str:
        return await self._batcher.submit(url, timeout_seconds)

    async def _run_batch(
        self, urls: list[str], timeout_seconds: int
    ) -> list[str | Exception]:
        session = await self._pool.acquire()
        broken = False
        try:
            config = build_scrape_config(self.settings, urls, timeout_seconds)
            return await run_scrape_in_session(session, self._base_dir, config)
        except BaseException:
            # A failed run may leave the session in a bad state; recycle it.
            broken = True
            raise
        finally:
            await self._pool.release(session, broken=broken, requests=len(urls))

    async def close(self) -> None:
        await self._batcher.close()
        await self._pool.close()
