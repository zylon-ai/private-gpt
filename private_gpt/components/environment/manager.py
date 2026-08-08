from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from private_gpt.components.environment.distributed import DistributedCoordinator
from private_gpt.components.environment.environment import Environment
from private_gpt.components.sandbox.mount import Mount

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    from private_gpt.components.environment.content_mounter import ContentMounter
    from private_gpt.components.environment.mounter import LayoutMounter
    from private_gpt.components.sandbox.base import SandboxProvider, SandboxSession

logger = logging.getLogger(__name__)

# Fraction of TTL remaining below which we proactively renew the sandbox.
_RENEW_THRESHOLD = 1 / 3

# Minimum seconds between two renewal attempts for the same session.
# Prevents rapid concurrent requests from all issuing a renewal.
_RENEW_SKIP_WINDOW = 30


class EnvironmentManager:
    """Owns the lifecycle of managed environments, keyed by session id.

    acquire() returns the live environment for a session, reusing it when the
    requested mounts and sandbox env are unchanged, or killing and recreating
    it otherwise. release() drops an environment and kills its backend sandbox.
    A background reaper kills environments idle past the TTL, and stale
    environments (e.g. after a backend server restart) are evicted and
    recreated on the next acquire(). Cross-process races on the same session
    are serialised with a per-session lock (Redis, with an in-memory
    fallback); the reaper also consults a shared last-activity clock.
    """

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        layout_mounter: LayoutMounter,
        content_mounters: list[ContentMounter],
        ttl_seconds: int,
        reaper_interval_seconds: int | None = None,
        *,
        coordinator: DistributedCoordinator | None = None,
    ) -> None:
        self._provider = sandbox_provider
        self._layout = layout_mounter
        self._content_mounters = content_mounters
        self._ttl = ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._coordinator = coordinator or DistributedCoordinator()
        self._active: dict[str, Environment] = {}
        self._lock = asyncio.Lock()
        self._creation_locks: dict[str, asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def acquire(
        self,
        session_id: str,
        mounts: list[Mount] | None = None,
        sandbox_env: dict[str, str] | None = None,
    ) -> Environment:
        # Serialize per session_id so concurrent calls cannot race into
        # creating two backend sandboxes for the same session (one would leak).
        creation_lock = await self._creation_lock(session_id)
        # Cross-process serialisation: only one worker may create/restore/
        # kill the session's container at a time.
        async with creation_lock, self._coordinator.session_lock(session_id) as locked:
            if not locked:
                logger.warning(
                    "Could not acquire distributed lock for session %s; "
                    "proceeding with local coordination only",
                    session_id,
                )
            return await self._acquire_locked(session_id, mounts, sandbox_env)

    async def _acquire_locked(
        self,
        session_id: str,
        mounts: list[Mount] | None,
        sandbox_env: dict[str, str] | None,
    ) -> Environment:
        async with self._lock:
            env = self._active.get(session_id)

        if env:
            if env._stale:
                # Sandbox died (e.g. server restart): discard it and
                # create a fresh one for this request.
                logger.warning(
                    "Sandbox for session %s is stale, recreating", session_id
                )
                async with self._lock:
                    self._active.pop(session_id, None)
                await self._kill(env.sandbox, session_id)
                return await self._create(
                    session_id, mounts, sandbox_env, force_new=True
                )

            elif self._mounts_changed(env, mounts, sandbox_env):
                # Mounts changed: kill the old sandbox and create a new one
                # so the new mounts are wired at container creation.
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
                    session_id, mounts, sandbox_env, force_new=True
                )

            else:
                env.touch()
                if not await self._maybe_renew(env, session_id):
                    # Renewal failed — discard the dead sandbox so this
                    # request does not run on it.
                    logger.warning(
                        "Sandbox for session %s could not be renewed, recreating",
                        session_id,
                    )
                    async with self._lock:
                        self._active.pop(session_id, None)
                    await self._kill(env.sandbox, session_id)
                    return await self._create(
                        session_id, mounts, sandbox_env, force_new=True
                    )

                if mounts:
                    # Content that could not be bind-mounted at creation is
                    # materialised lazily just before the first exec().
                    env.add_pending(mounts)
                    await env._flush_pending()
                return env

        return await self._create(session_id, mounts, sandbox_env)

    def release(self, session_id: str) -> None:
        """Drop the environment and release its backend resources."""
        env = self._active.pop(session_id, None)
        self._creation_locks.pop(session_id, None)
        if env:
            self._spawn(
                self._release_and_kill(session_id, env),
                f"kill sandbox on release ({session_id})",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _release_and_kill(self, session_id: str, env: Environment) -> None:
        """Kill the released sandbox unless a newer env took over meanwhile."""
        async with self._coordinator.session_lock(session_id):
            async with self._lock:
                current = self._active.get(session_id)
            if current is not None and current is not env:
                # A newer environment was created for this session while we
                # waited for the lock — do not kill it.
                logger.info(
                    "Skipping kill on release for session %s (newer env active)",
                    session_id,
                )
                return
            await self._kill(env.sandbox, session_id)

    async def _maybe_renew(self, env: Environment, session_id: str) -> bool:
        """Renew the sandbox's lifetime if it is approaching expiry.

        Returns True when the sandbox is still usable; False when the renewal
        failed and the caller must discard the sandbox and create a fresh one.

        Renewal is skipped when another renewal happened recently
        (within ``_RENEW_SKIP_WINDOW`` seconds) to avoid a storm of
        renewal calls when many requests arrive simultaneously.
        """
        now = time.monotonic()
        age = now - env.ttl_start
        remaining = self._ttl - age
        if remaining >= self._ttl * _RENEW_THRESHOLD:
            return True  # plenty of time left

        since_last_renew = now - env.last_renewed
        if since_last_renew < _RENEW_SKIP_WINDOW:
            return True  # a renewal was issued very recently; skip

        logger.info(
            "Sandbox for session %s has ~%.0fs remaining (TTL %ds), renewing",
            session_id,
            remaining,
            self._ttl,
        )
        try:
            await self._provider.renew_session(env.sandbox)
            env.ttl_start = now
            env.last_renewed = now
        except Exception as exc:
            logger.warning(
                "Failed to renew sandbox for session %s: %s", session_id, exc
            )
            return False
        return True

    async def _create(
        self,
        session_id: str,
        mounts: list[Mount] | None = None,
        sandbox_env: dict[str, str] | None = None,
        *,
        force_new: bool = False,
    ) -> Environment:
        await asyncio.to_thread(self._layout.ensure_ready)

        # Layout volumes (workspace, uploads, outputs).
        layout_volumes = self._layout.session_volumes(session_id)
        requested_targets = {mount.target for mount in mounts or []}
        # An explicit mount is allowed to replace a layout mount at the same
        # target (for example a durable file replacing a session file). This
        # is generic mount precedence, not artifact knowledge.
        volumes = [
            volume
            for volume in (layout_volumes or [])
            if volume.target not in requested_targets
        ]

        # Mount specs — always added for writability enforcement.
        specs = [
            spec
            for spec in self._layout.mount_specs()
            if spec.target not in requested_targets
        ]
        for mount in mounts or []:
            specs.append(Mount(target=mount.target, access=mount.access))

        # Mounts that support eager volume-mounting (a resolved source dir, or
        # a storage-backed bundle whose host folder is already present).
        # Pre-populate _mounted so they skip materialize().
        pre_mounted: set[str] = set()
        seen_volume_names: set[str] = set()
        for mount in mounts or []:
            if mount.host_path is not None:
                volumes.append(mount)
                pre_mounted.add(mount.target)
                continue
            mounter = self._find_content_mounter(mount)
            if mounter:
                vol = await mounter.prepare_volume(mount, session_id)
                if vol:
                    if vol.name not in seen_volume_names:
                        volumes.append(vol)
                        seen_volume_names.add(vol.name)
                    pre_mounted.add(mount.target)

        # Fingerprint of everything that would change the container: the
        # requested mounts and the injected env. Stored in sandbox metadata at
        # creation so a restore from another pod can detect stale containers.
        fingerprint = self._fingerprint(mounts or [], sandbox_env)

        if force_new:
            sandbox = None
        else:
            sandbox = await self._provider.restore_session(
                session_id,
                timeout=self._ttl,
                bundle_specs=specs,
                fingerprint=fingerprint,
            )
        if sandbox is None:
            sandbox = await self._provider.create_session(
                timeout=self._ttl,
                bundle_specs=specs,
                session_id=session_id,
                volumes=volumes or None,
                env=sandbox_env,
                fingerprint=fingerprint,
            )

        try:
            # Layout dirs are only needed when not volume-backed.
            if layout_volumes is None:
                await asyncio.gather(
                    *[sandbox.make_dir(m.target) for m in self._layout.layout]
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
            workspace=self._layout.workspace_target,
            content_mounters=self._content_mounters,
            owner=self._coordinator.instance_id,
            activity_sink=self._coordinator.set_activity,
        )
        env._mounted.update(pre_mounted)
        # Record the mounts this env was created with, so acquire() can detect
        # mount changes on later reuses (and recreate instead of materialising).
        env._mount_keys = self._mount_keys(mounts or [])
        env._sandbox_env = dict(sandbox_env or {})

        # Deferred mounts: not volume-mounted, will be materialised on exec().
        deferred = [m for m in (mounts or []) if m.target not in pre_mounted]
        env.add_pending(deferred)

        async with self._lock:
            self._active[session_id] = env

        self._ensure_reaper()
        return env

    def _find_content_mounter(self, mount: Mount) -> ContentMounter | None:
        return next((m for m in self._content_mounters if m.can_handle(mount)), None)

    @staticmethod
    def _fingerprint(mounts: list[Mount], sandbox_env: dict[str, str] | None) -> str:
        """Stable, cross-process fingerprint of requested mounts + env.

        Must be byte-identical on every pod for the same input so it can be
        compared against the value stored in sandbox metadata at creation.
        """
        keys = sorted(
            (
                m.target,
                m.access,
                str(m.host_path) if m.host_path is not None else "",
                m.uri_source.cache_key if m.uri_source is not None else (),
                m.etag or "",
            )
            for m in mounts
        )
        return json.dumps(
            {"mounts": keys, "env": sorted((sandbox_env or {}).items())},
            sort_keys=True,
        )

    @staticmethod
    def _mount_keys(mounts: list[Mount]) -> frozenset[tuple[object, ...]]:
        """Identity of each requested mount: target + access + source + storage prefix.

        The storage prefix (when present) distinguishes content versions that
        share a canonical mount path; the etag captures content-level changes.
        """
        return frozenset(
            (
                m.target,
                m.access,
                str(m.host_path) if m.host_path is not None else "",
                m.uri_source.cache_key if m.uri_source is not None else (),
                m.etag or "",
            )
            for m in mounts
        )

    def _mounts_changed(
        self,
        env: Environment,
        mounts: list[Mount] | None,
        sandbox_env: dict[str, str] | None,
    ) -> bool:
        """True when the requested mounts differ from the live env's mounts."""
        if self._mount_keys(mounts or []) != env._mount_keys:
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
        """Kill sandboxes idle past the TTL.

        Kills sandboxes that are idle locally and on the shared last-activity clock.
        """
        now_mono = time.monotonic()
        now_wall = time.time()
        expired: list[tuple[str, Environment]] = []
        async with self._lock:
            for session_id, env in list(self._active.items()):
                if env.idle_seconds(now_mono) <= self._ttl:
                    continue
                shared = await self._coordinator.get_activity(session_id)
                if shared is not None and (now_wall - shared) <= self._ttl:
                    # Active on another pod/worker — leave it alone.
                    continue
                self._active.pop(session_id, None)
                self._creation_locks.pop(session_id, None)
                expired.append((session_id, env))

        for session_id, env in expired:
            self._spawn(
                self._kill(env.sandbox, session_id),
                f"kill idle sandbox ({session_id})",
            )
