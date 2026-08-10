from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from private_gpt.components.environment.distributed import DistributedCoordinator
from private_gpt.components.environment.environment import Environment
from private_gpt.components.sandbox.mount import Mount

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    from private_gpt.components.environment.mounter import LayoutMounter
    from private_gpt.components.sandbox.base import SandboxProvider, SandboxSession
    from private_gpt.settings.settings import NamespaceConfig

logger = logging.getLogger(__name__)

_RENEW_THRESHOLD = 1 / 3

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

    Every mount is a bind volume wired at container creation — there is no
    lazy materialization into a running sandbox. The full volume set is:

    1. one volume per configured filesystem namespace at its canonical
       container root (``/mnt/{name}/``), even when empty,
    2. the session layout volumes (workspace, uploads, outputs),
    3. the requested content mounts (folders or files, exact targets).
    """

    def __init__(
        self,
        sandbox_provider: SandboxProvider,
        layout_mounter: LayoutMounter,
        ttl_seconds: int,
        reaper_interval_seconds: int | None = None,
        *,
        coordinator: DistributedCoordinator | None = None,
        namespaces: dict[str, NamespaceConfig] | None = None,
    ) -> None:
        self._provider = sandbox_provider
        self._layout = layout_mounter
        self._ttl = ttl_seconds
        self._reaper_interval = reaper_interval_seconds
        self._coordinator = coordinator or DistributedCoordinator()
        self._namespaces = namespaces or {}
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
            if self._mounts_changed(env, mounts, sandbox_env):
                logger.info(
                    "Mounts changed for session %s, recreating sandbox",
                    session_id,
                )
                async with self._lock:
                    self._active.pop(session_id, None)
                await self._kill(env.sandbox, session_id)
                return await self._create(
                    session_id, mounts, sandbox_env, force_new=True
                )

            env.touch()
            if not await self._maybe_renew(env, session_id):
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

        mounts = mounts or []

        layout_volumes = self._layout.session_volumes(session_id)
        volumes = _dedupe_volumes(
            self._namespace_volumes()
            + (layout_volumes or [])
            + [m for m in mounts if m.host_path is not None]
        )

        specs = self._layout.mount_specs()
        specs.extend(Mount(target=m.target, access=m.access) for m in mounts)

        fingerprint = self._fingerprint(mounts, sandbox_env)

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
            owner=self._coordinator.instance_id,
            activity_sink=self._coordinator.set_activity,
        )
        env._mount_keys = self._mount_keys(mounts)
        env._sandbox_env = dict(sandbox_env or {})

        async with self._lock:
            self._active[session_id] = env

        self._ensure_reaper()
        return env

    def _namespace_volumes(self) -> list[Mount]:
        """One bind volume per configured namespace root at ``/mnt/{name}/``.

        Root volumes are mounted even when empty so the agent can write inside
        the namespace and the runtime never has to create those paths inside
        the container (which fails for provider-managed roots like /mnt).
        """
        volumes: list[Mount] = []
        for name, config in sorted(self._namespaces.items()):
            if not config.root:
                continue
            host = Path(config.root)
            host.mkdir(parents=True, exist_ok=True)
            volumes.append(
                Mount(
                    name=f"namespace-{name}",
                    target=f"/mnt/{name}/",
                    access=config.default_mode,
                    host_path=host,
                )
            )
        return volumes

    @staticmethod
    def _fingerprint(mounts: list[Mount], sandbox_env: dict[str, str] | None) -> str:
        """Stable, cross-process fingerprint of requested mounts + env.

        Must be byte-identical on every pod for the same input so it can be
        compared against the value stored in sandbox metadata at creation.
        Signed URIs are deliberately absent — they rotate every request and
        are not mount identity.
        """
        keys = sorted(_mount_identity(m) for m in mounts)
        return json.dumps(
            {"mounts": keys, "env": sorted((sandbox_env or {}).items())},
            sort_keys=True,
        )

    @staticmethod
    def _mount_keys(mounts: list[Mount]) -> frozenset[tuple[object, ...]]:
        """Identity of each requested mount: target + access + source.

        Storage identity (namespace/scope/path + host path) and the etag
        distinguish content versions that share a canonical mount target.
        """
        return frozenset(_mount_identity(m) for m in mounts)

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
                    continue
                self._active.pop(session_id, None)
                self._creation_locks.pop(session_id, None)
                expired.append((session_id, env))

        for session_id, env in expired:
            self._spawn(
                self._kill(env.sandbox, session_id),
                f"kill idle sandbox ({session_id})",
            )


def _mount_identity(mount: Mount) -> tuple[object, ...]:
    """Stable identity of one mount: target + access + storage source + etag.

    The URI is a hydration origin, not identity: signed URLs rotate every
    request and must not cause spurious sandbox recreations.
    """
    source = mount.source
    return (
        mount.target,
        mount.access,
        str(mount.host_path) if mount.host_path is not None else "",
        source.namespace if source else "",
        source.scope if source else "",
        source.path if source else "",
        mount.etag or "",
    )


def _dedupe_volumes(volumes: list[Mount]) -> list[Mount]:
    """Collapse identical bind volumes; reject same-target different-source.

    Mirrors Docker: the same ``(host_path, target, mode)`` is idempotent,
    while two different sources claiming the exact same target is ambiguous
    and must fail before any sandbox is created.

    Also guarantees volume names stay unique: duplicate volume names are
    rejected by sandbox backends, so any collision is resolved by suffixing
    a short target hash.
    """
    seen_sources: dict[str, Mount] = {}
    merged: list[Mount] = []
    for volume in volumes:
        if volume.host_path is None:
            continue
        target = volume.target.rstrip("/") or "/"
        if target in seen_sources:
            other = seen_sources[target]
            if (
                str(other.host_path) == str(volume.host_path)
                and other.access == volume.access
            ):
                continue  # identical bind — idempotent
            raise ValueError(
                "Conflicting sandbox mount targets: "
                f"{other.target!r} ({other.host_path}) and "
                f"{volume.target!r} ({volume.host_path})"
            )
        seen_sources[target] = volume
        merged.append(volume)

    used_names: set[str] = set()
    for volume in merged:
        name = volume.name or _target_volume_name(volume.target)
        while name in used_names:
            name = (
                f"{name}-{hashlib.sha1(volume.target.encode('utf-8')).hexdigest()[:8]}"
            )
        used_names.add(name)
        volume.name = name
    return merged


def _target_volume_name(target: str) -> str:
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:16]
    return f"mount-{digest}"
