from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from private_gpt.components.environment.environment import Environment
from private_gpt.components.sandbox.mount import MountSpec

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

    ``renew_on_acquire`` switches the backend lifetime refresh from the reaper
    timer to an eager refresh on every acquire() — i.e. every new message that
    reaches a tool. The reaper then only reaps idle sandboxes.

    ``recreate_on_mount_change`` kills the old sandbox and creates a new one
    whenever the requested mounts (bundles, extra volumes, sandbox env) differ
    from what the live env was created with. Recreating wires the new mounts at
    container creation (fastest path) instead of materializing/copying files
    into a running container.

    A stale sandbox (e.g. after the backend server restarts) marks the
    Environment as _stale during the first failing exec/flush; acquire() then
    evicts and recreates transparently.

    A background reaper kills environments idle past the TTL.
    """

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        layout_mounter: LayoutMounter,
        content_mounters: list[ContentMounter],
        ttl_seconds: int,
        reaper_interval_seconds: int | None = None,
        *,
        renew_on_acquire: bool = False,
        recreate_on_mount_change: bool = False,
    ) -> None:
        self._provider = sandbox_provider
        self._layout = layout_mounter
        self._content_mounters = content_mounters
        self._ttl = ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._renew_on_acquire = renew_on_acquire
        self._recreate_on_mount_change = recreate_on_mount_change
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
        sandbox_env: dict[str, str] | None = None,
        extra_volumes: list[MountSpec] | None = None,
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
                elif self._recreate_on_mount_change and self._mounts_changed(
                    env, extra_bundles, bundles_to_remove, sandbox_env, extra_volumes
                ):
                    # Mounts changed: kill the old sandbox and create a new one
                    # so the new mounts are wired at container creation (no
                    # materialization / file copy into a running container).
                    logger.info(
                        "Mounts changed for session %s, recreating sandbox",
                        session_id,
                    )
                    async with self._lock:
                        self._active.pop(session_id, None)
                    # Kill synchronously so restore_session() cannot rediscover
                    # the old sandbox and reconnect with old mounts.
                    await self._kill(env.sandbox, session_id)
                    return await self._create(
                        session_id,
                        extra_bundles,
                        bundles_to_remove,
                        sandbox_env,
                        extra_volumes,
                        force_new=True,
                    )
                else:
                    env.touch()
                    if self._renew_on_acquire:
                        # Eager refresh: a new message arrived, extend the
                        # backend lifetime now instead of on the reaper timer.
                        try:
                            await self._provider.renew_session(env.sandbox)
                        except Exception as exc:
                            logger.warning(
                                "Failed to eagerly renew sandbox for session %s, "
                                "recreating: %s",
                                session_id,
                                exc,
                            )
                            async with self._lock:
                                self._active.pop(session_id, None)
                            await self._kill(env.sandbox, session_id)
                            return await self._create(
                                session_id,
                                extra_bundles,
                                bundles_to_remove,
                                sandbox_env,
                                extra_volumes,
                                force_new=True,
                            )
                    if bundles_to_remove:
                        await env.remove_bundles(bundles_to_remove)
                    if extra_bundles:
                        # Container is already running — push bundles immediately
                        # so skills are accessible before the next exec().
                        env.add_pending(extra_bundles)
                        await env._flush_pending()
                    return env
            return await self._create(
                session_id, extra_bundles, bundles_to_remove, sandbox_env, extra_volumes
            )

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
        sandbox_env: dict[str, str] | None = None,
        extra_volumes: list[MountSpec] | None = None,
        *,
        force_new: bool = False,
    ) -> Environment:
        await asyncio.to_thread(self._layout.ensure_ready)

        # Layout volumes (workspace, uploads, outputs).
        layout_volumes = self._layout.session_volumes(session_id)
        volumes = list(layout_volumes or [])
        # Artifact mount refs from the Backend mount plan.
        volumes.extend(extra_volumes or [])

        # Bundle mount specs — always added for writability enforcement.
        specs = self._layout.mount_specs()
        for bundle in extra_bundles or []:
            specs.append(
                MountSpec(canonical=bundle.canonical_path, writable=bundle.writable)
            )

        # Bundles that support eager volume-mounting (e.g. local storage,
        # S3FS bind-mount). Pre-populate _mounted so they skip materialize().
        # Deduplicate by volume name: multiple skills from the same collection
        # return the same collection-level MountSpec; only the first is added.
        pre_mounted: set[str] = set()
        seen_volume_names: set[str] = set()
        for bundle in extra_bundles or []:
            mounter = self._find_content_mounter(bundle)
            if mounter:
                vol = await mounter.prepare_volume(bundle, session_id)
                if vol:
                    if vol.name not in seen_volume_names:
                        volumes.append(vol)
                        seen_volume_names.add(vol.name)
                    pre_mounted.add(bundle.canonical_path)

        if force_new:
            sandbox = None
        else:
            sandbox = await self._provider.restore_session(
                session_id, timeout=self._ttl, bundle_specs=specs
            )
        if sandbox is None:
            sandbox = await self._provider.create_session(
                timeout=self._ttl,
                bundle_specs=specs,
                session_id=session_id,
                volumes=volumes or None,
                env=sandbox_env,
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
        # Record the mounts this env was created with, so acquire() can detect
        # mount changes on later reuses (and recreate instead of materializing).
        env._bundle_paths = self._bundle_keys(extra_bundles or [])
        env._volume_keys = self._volume_keys(extra_volumes or [])
        env._sandbox_env = dict(sandbox_env or {})

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

    @staticmethod
    def _bundle_keys(
        bundles: list[ContentBundle],
    ) -> frozenset[tuple[str, str]]:
        """Identity of each requested bundle: mount path + storage prefix.

        The storage prefix (when present, e.g. StoredBundle) distinguishes
        content versions that share a canonical mount path.
        """
        return frozenset(
            (b.canonical_path, getattr(b, "storage_prefix", "")) for b in bundles
        )

    @staticmethod
    def _volume_keys(volumes: list[MountSpec]) -> frozenset[tuple[object, ...]]:
        """Identity of each extra volume: name + host path + canonical + mode."""
        return frozenset(
            (v.name, str(v.host_path), v.canonical, v.read_only) for v in volumes
        )

    def _mounts_changed(
        self,
        env: Environment,
        extra_bundles: list[ContentBundle] | None,
        bundles_to_remove: list[str] | None,
        sandbox_env: dict[str, str] | None,
        extra_volumes: list[MountSpec] | None,
    ) -> bool:
        """True when the requested mounts differ from the live env's mounts."""
        requested_bundles = self._bundle_keys(extra_bundles or [])
        if requested_bundles != env._bundle_paths:
            return True
        if bundles_to_remove and any(
            p in {path for path, _ in env._bundle_paths} for p in bundles_to_remove
        ):
            return True
        if self._volume_keys(extra_volumes or []) != env._volume_keys:
            return True
        return dict(sandbox_env or {}) != env._sandbox_env

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

        # With eager refresh on acquire, the backend lifetime is extended per
        # new message — no periodic renewal on the reaper timer.
        if not self._renew_on_acquire:
            for env in live:
                try:
                    await self._provider.renew_session(env.sandbox)
                except Exception as exc:
                    logger.warning(
                        "Failed to renew sandbox for session %s: %s", env.id, exc
                    )
