from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from private_gpt.components.environment.environment import Environment

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    from private_gpt.components.environment.mounter import Mounter
    from private_gpt.components.sandbox.base import SandboxProvider, SandboxSession
    from private_gpt.components.sandbox.content_bundle import ContentBundle

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Owns the lifecycle of managed environments, keyed by session id.

    Generic over the sandbox backend (SandboxProvider) and the mount strategy
    (Mounter). acquire() reuses a live environment, restores a backend sandbox
    when the provider supports it, or creates a fresh one. A background reaper
    kills environments idle past the TTL and renews the backend lifetime of
    the ones still in use, so a long conversation is never cut off mid-flight.
    """

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        mounter: Mounter,
        ttl_seconds: int,
        reaper_interval_seconds: int,
    ) -> None:
        self._provider = sandbox_provider
        self._mounter = mounter
        self._ttl = ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._active: dict[str, Environment] = {}
        self._lock = asyncio.Lock()
        self._creation_locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._mounter.ensure_ready()

    async def acquire(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> Environment:
        # Serialize per session_id so concurrent calls cannot race into
        # creating two backend sandboxes for the same session (one would leak).
        creation_lock = await self._creation_lock(session_id)
        async with creation_lock:
            async with self._lock:
                env = self._active.get(session_id)
            if env:
                env.touch()
                if extra_bundles:
                    # Content activated mid-session (e.g. a newly enabled
                    # skill) must appear in the live sandbox. Idempotent.
                    await self._mounter.prepare_bundles(env.sandbox, extra_bundles)
                return env
            return await self._create(session_id, extra_bundles)

    def release(self, session_id: str) -> None:
        """Drop the environment and release its backend resources."""
        env = self._active.pop(session_id, None)
        self._creation_locks.pop(session_id, None)
        if env:
            self._spawn(
                self._kill(env.sandbox, session_id),
                f"kill sandbox on release ({session_id})",
            )

    async def _create(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None,
    ) -> Environment:
        specs = self._mounter.mount_specs(extra_bundles)

        sandbox = await self._provider.restore_session(
            session_id, timeout=self._ttl, bundle_specs=specs
        )
        if sandbox is None:
            sandbox = await self._provider.create_session(
                timeout=self._ttl,
                bundle_specs=specs,
                session_id=session_id,
                volumes=self._mounter.session_volumes(session_id, extra_bundles),
            )

        try:
            await self._mounter.prepare(sandbox, session_id, extra_bundles)
        except Exception:
            # A half-initialized sandbox must not be reused or leaked.
            self._spawn(
                self._kill(sandbox, session_id),
                f"kill sandbox after failed mount setup ({session_id})",
            )
            raise

        env = Environment(
            id=session_id,
            sandbox=sandbox,
            workspace=self._mounter.workspace_canonical,
        )
        async with self._lock:
            self._active[session_id] = env

        self._ensure_reaper()
        return env

    async def _creation_lock(self, session_id: str) -> asyncio.Lock:
        async with self._lock:
            lock = self._creation_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._creation_locks[session_id] = lock
            return lock

    async def _kill(self, sandbox: SandboxSession, session_id: str) -> None:
        try:
            await self._provider.kill_session(sandbox, session_id)
            logger.info("Killed sandbox for session %s", session_id)
        except Exception as exc:
            # Backend-side TTL is the backstop if the kill never lands.
            logger.warning("Failed to kill sandbox for session %s: %s", session_id, exc)

    def _spawn(self, coro: Coroutine[Any, Any, Any], what: str) -> None:
        """Run a fire-and-forget coroutine, keeping a strong reference.

        Bare ``create_task`` results can be garbage-collected mid-flight;
        tracking them in a set guarantees completion and surfaces errors.
        """
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            logger.warning("No running event loop — skipped: %s", what)
            return
        self._background_tasks.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error("Background task failed (%s): %s", what, t.exception())

        task.add_done_callback(_done)

    def _ensure_reaper(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.get_running_loop().create_task(
                self._reaper_loop()
            )

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reaper_interval)
            try:
                await self._reap_once()
            except Exception:
                logger.exception("Environment reaper iteration failed")

    async def _reap_once(self) -> None:
        now = time.monotonic()
        expired: list[tuple[str, Environment]] = []
        live: list[Environment] = []
        async with self._lock:
            for session_id, env in list(self._active.items()):
                if env.idle_seconds(now) > self._ttl:
                    self._active.pop(session_id, None)
                    self._creation_locks.pop(session_id, None)
                    expired.append((session_id, env))
                else:
                    live.append(env)

        for session_id, env in expired:
            self._spawn(
                self._kill(env.sandbox, session_id),
                f"kill idle sandbox ({session_id})",
            )

        # Keep busy environments alive: backends may expire them ttl_seconds
        # after creation unless renewed.
        for env in live:
            try:
                await self._provider.renew_session(env.sandbox)
            except Exception as exc:
                logger.warning(
                    "Failed to renew sandbox for session %s: %s", env.id, exc
                )
