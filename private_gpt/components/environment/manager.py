from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from private_gpt.components.environment.environment import Environment
from private_gpt.components.sandbox.mount import SandboxMountSpec

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    from private_gpt.components.environment.content_mounter import ContentMounter
    from private_gpt.components.environment.mounter import LayoutMounter
    from private_gpt.components.sandbox.base import SandboxProvider, SandboxSession
    from private_gpt.components.sandbox.content_bundle import ContentBundle

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Owns the lifecycle of managed environments, keyed by session id.

    Generic over the sandbox backend (SandboxProvider), the layout strategy
    (LayoutMounter), and the ordered list of content mounters (ContentMounter).

    acquire() reuses a live environment, restores a backend sandbox when the
    provider supports it, or creates a fresh one. Bundle registration on reuse
    is zero-network: bundles are added to a pending list and materialized lazily
    just before the first exec() that follows.

    A stale sandbox (e.g. after the backend server restarts) marks the
    Environment as _stale during the first failing exec/flush; acquire() then
    evicts and recreates transparently.

    A background reaper kills environments idle past the TTL and renews the
    backend lifetime of those still in use.
    """

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        layout_mounter: LayoutMounter,
        content_mounters: list[ContentMounter],
        ttl_seconds: int,
        reaper_interval_seconds: int | None = None,
    ) -> None:
        self._provider = sandbox_provider
        self._layout = layout_mounter
        self._content_mounters = content_mounters
        self._ttl = ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._active: dict[str, Environment] = {}
        self._lock = asyncio.Lock()
        self._creation_locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def acquire(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
        bundles_to_remove: list[str] | None = None,
    ) -> Environment:
        # Serialize per session_id so concurrent calls cannot race into
        # creating two backend sandboxes for the same session (one would leak).
        creation_lock = await self._creation_lock(session_id)
        async with creation_lock:
            async with self._lock:
                env = self._active.get(session_id)
            if env:
                if env._stale:
                    # Sandbox died (e.g. server restart). Evict and fall
                    # through to _create() so the next acquire gets a fresh env.
                    logger.warning(
                        "Sandbox for session %s is stale, recreating", session_id
                    )
                    async with self._lock:
                        self._active.pop(session_id, None)
                    self._spawn(
                        self._kill(env.sandbox, session_id),
                        f"kill stale sandbox ({session_id})",
                    )
                else:
                    env.touch()
                    if bundles_to_remove:
                        await env.remove_bundles(bundles_to_remove)
                    if extra_bundles:
                        # Zero network calls: bundles are registered as pending
                        # and materialized lazily before the next exec().
                        env.add_pending(extra_bundles)
                    return env
            return await self._create(session_id, extra_bundles, bundles_to_remove)

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
        bundles_to_remove: list[str] | None = None,
    ) -> Environment:
        await asyncio.to_thread(self._layout.ensure_ready)

        # Layout volumes (workspace, uploads, outputs).
        layout_volumes = self._layout.session_volumes(session_id)
        volumes = list(layout_volumes or [])

        # Bundle mount specs — always added for writability enforcement.
        specs = self._layout.mount_specs()
        for bundle in extra_bundles or []:
            specs.append(
                SandboxMountSpec(
                    canonical=bundle.canonical_path, writable=bundle.writable
                )
            )

        # Bundles that support eager volume-mounting (e.g. local storage,
        # S3FS bind-mount). Pre-populate _mounted so they skip materialize().
        pre_mounted: set[str] = set()
        for bundle in extra_bundles or []:
            mounter = self._find_content_mounter(bundle)
            if mounter:
                vol = await mounter.prepare_volume(bundle, session_id)
                if vol:
                    volumes.append(vol)
                    pre_mounted.add(bundle.canonical_path)

        sandbox = await self._provider.restore_session(
            session_id, timeout=self._ttl, bundle_specs=specs
        )
        if sandbox is None:
            sandbox = await self._provider.create_session(
                timeout=self._ttl,
                bundle_specs=specs,
                session_id=session_id,
                volumes=volumes or None,
            )

        try:
            # Layout dirs are only needed when not volume-backed.
            if layout_volumes is None:
                await asyncio.gather(
                    *[sandbox.make_dir(m.canonical) for m in self._layout.layout]
                )
        except Exception:
            self._spawn(
                self._kill(sandbox, session_id),
                f"kill sandbox after failed layout setup ({session_id})",
            )
            raise

        env = Environment(
            id=session_id,
            sandbox=sandbox,
            workspace=self._layout.workspace_canonical,
            content_mounters=self._content_mounters,
        )
        env._mounted.update(pre_mounted)

        if bundles_to_remove:
            await env.remove_bundles(bundles_to_remove)

        # Deferred bundles: not volume-mounted, will be materialized on exec().
        deferred = [
            b for b in (extra_bundles or []) if b.canonical_path not in pre_mounted
        ]
        env.add_pending(deferred)

        async with self._lock:
            self._active[session_id] = env

        self._ensure_reaper()
        return env

    def _find_content_mounter(self, bundle: ContentBundle) -> ContentMounter | None:
        return next((m for m in self._content_mounters if m.can_handle(bundle)), None)

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
        if not self._reaper_interval:
            return

        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.get_running_loop().create_task(
                self._reaper_loop()
            )

    async def _reaper_loop(self) -> None:
        if not self._reaper_interval:
            return

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
