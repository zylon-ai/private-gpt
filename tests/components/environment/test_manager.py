"""Tests for EnvironmentManager mount-change recreation and on-acquire renewal."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from private_gpt.components.environment import distributed as _dist
from private_gpt.components.environment.manager import (
    _RENEW_THRESHOLD,
    EnvironmentManager,
)
from private_gpt.components.sandbox.mount import Mount, UriSource


@pytest.fixture(autouse=True)
def _clean_distributed_state() -> None:
    """Isolate the process-wide fallback coordination state per test."""
    _dist._fallback_locks.clear()
    _dist._fallback_activity.clear()
    yield
    _dist._fallback_locks.clear()
    _dist._fallback_activity.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLayout:
    """Layout mounter returning no host volumes (dirs live in the sandbox)."""

    layout = ()
    workspace_target = "/home/agent/workspace/"

    def ensure_ready(self) -> None:
        pass

    def session_volumes(self, session_id: str) -> list[Mount] | None:
        return None

    def mount_specs(self) -> list[Mount]:
        return []


class FakeSandbox:
    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.killed = False
        self.mkdir_calls: list[str] = []

    async def make_dir(self, canonical: str, **kwargs) -> None:
        self.mkdir_calls.append(canonical)

    async def remove_mount(self, canonical: str) -> None:
        pass


class FakeProvider:
    """Records lifecycle calls; restore_session always fails (fresh create)."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.killed: list[str] = []
        self.renewed: list[str] = []
        self.fingerprints: list[str | None] = []
        self._counter = 0

    async def restore_session(
        self, session_id, timeout=None, bundle_specs=None, *, fingerprint=None
    ):
        return None

    async def create_session(
        self,
        user_id=None,
        timeout=None,
        bundle_specs=None,
        *,
        session_id=None,
        volumes=None,
        env=None,
        fingerprint=None,
    ) -> FakeSandbox:
        self._counter += 1
        sandbox_id = f"sandbox-{self._counter}"
        self.created.append(sandbox_id)
        self.fingerprints.append(fingerprint)
        return FakeSandbox(sandbox_id)

    async def renew_session(self, session) -> None:
        self.renewed.append(session.sandbox_id)

    async def kill_session(self, session, session_id=None) -> None:
        session.killed = True
        self.killed.append(session.sandbox_id)

    async def delete_session(self, session) -> None:
        await self.kill_session(session)


def _manager(**kwargs) -> EnvironmentManager:
    kwargs.setdefault("sandbox_provider", FakeProvider())
    return EnvironmentManager(
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
        **kwargs,
    )


def _bundle_mount(canonical: str, prefix: str = "") -> Mount:
    """A storage-backed skill mount."""
    return Mount(
        name=f"skill:{canonical}",
        target=canonical,
        access="ro",
        uri_source=UriSource(uri=prefix or canonical, fetch=lambda: []),
    )


def _volume(name: str, canonical: str) -> Mount:
    return Mount(name=name, target=canonical, host_path=Path("/tmp") / name)


async def _sleep_tasks(manager: EnvironmentManager) -> None:
    """Let fire-and-forget background tasks (kills) finish."""
    tasks = list(manager._background_tasks)
    for task in tasks:
        await task
    manager._background_tasks.clear()


# ---------------------------------------------------------------------------
# Mount-change recreation (always enabled)
# ---------------------------------------------------------------------------


async def test_acquire_recreates_when_bundles_change() -> None:
    manager = _manager()

    first = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    await _sleep_tasks(manager)
    second = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    await _sleep_tasks(manager)

    assert first is not second
    assert len(manager._provider.created) == 2
    # The old sandbox was killed, not materialised into.
    assert manager._provider.killed == ["sandbox-1"]
    assert len(manager._provider.renewed) == 0


async def test_acquire_reuses_when_mounts_unchanged() -> None:
    manager = _manager()

    first = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    second = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    await _sleep_tasks(manager)

    assert first is second
    assert len(manager._provider.created) == 1
    assert manager._provider.killed == []


async def test_acquire_recreates_when_volumes_change() -> None:
    manager = _manager()

    first = await manager.acquire("s1", mounts=[_volume("v1", "/mnt/data/")])
    await _sleep_tasks(manager)
    second = await manager.acquire(
        "s1",
        mounts=[_volume("v1", "/mnt/data/"), _volume("v2", "/mnt/other/")],
    )
    await _sleep_tasks(manager)

    assert first is not second
    assert len(manager._provider.created) == 2
    assert manager._provider.killed == ["sandbox-1"]


async def test_acquire_recreates_when_env_changes() -> None:
    manager = _manager()

    first = await manager.acquire("s1", sandbox_env={"TOKEN": "a"})
    await _sleep_tasks(manager)
    second = await manager.acquire("s1", sandbox_env={"TOKEN": "b"})
    await _sleep_tasks(manager)

    assert first is not second
    assert len(manager._provider.created) == 2
    assert manager._provider.killed == ["sandbox-1"]


async def test_acquire_reuses_when_mounts_unchanged_again() -> None:
    manager = _manager()

    first = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    second = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    await _sleep_tasks(manager)

    assert first is second
    assert len(manager._provider.created) == 1


async def test_acquire_recreates_when_bundle_removed() -> None:
    manager = _manager()

    first = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    await _sleep_tasks(manager)
    second = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/")],
    )
    await _sleep_tasks(manager)

    assert first is not second
    assert manager._provider.killed == ["sandbox-1"]


# ---------------------------------------------------------------------------
# On-acquire throttled renewal (TTL threshold)
# ---------------------------------------------------------------------------


async def test_no_renewal_when_ttl_healthy() -> None:
    """No renewal when there is plenty of lifetime left."""
    manager = _manager()

    await manager.acquire("s1")
    await manager.acquire("s1")  # reuse — TTL is fresh, no renewal expected

    assert manager._provider.renewed == []


async def test_renewal_triggered_when_ttl_below_threshold() -> None:
    """Renewal fires when remaining TTL < 1/3 of configured TTL."""
    manager = _manager()
    ttl = manager._ttl  # 3600s

    env = await manager.acquire("s1")
    # Simulate the sandbox being old enough that < 1/3 TTL remains.
    env.ttl_start = time.monotonic() - ttl * (1 - _RENEW_THRESHOLD + 0.01)
    env.last_renewed = 0.0  # never renewed — skip-window not active

    await manager.acquire("s1")  # should trigger renewal

    assert manager._provider.renewed == ["sandbox-1"]


async def test_renewal_skipped_within_skip_window() -> None:
    """A second renewal is not issued if one happened very recently."""
    manager = _manager()
    ttl = manager._ttl

    env = await manager.acquire("s1")
    # Age the sandbox so it would normally renew.
    env.ttl_start = time.monotonic() - ttl * (1 - _RENEW_THRESHOLD + 0.01)
    env.last_renewed = 0.0

    # First reuse → renewal fires.
    await manager.acquire("s1")
    assert len(manager._provider.renewed) == 1

    # Second reuse immediately after → skip window blocks it.
    await manager.acquire("s1")
    assert len(manager._provider.renewed) == 1  # still just one renewal


async def test_stale_on_renew_failure_triggers_recreate() -> None:
    """A failed renewal discards the container and recreates immediately."""

    class FlakyProvider(FakeProvider):
        async def renew_session(self, session) -> None:
            raise RuntimeError("sandbox expired")

    provider = FlakyProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
    )
    ttl = manager._ttl

    env = await manager.acquire("s1")
    # Push env into renewal territory.
    env.ttl_start = time.monotonic() - ttl * (1 - _RENEW_THRESHOLD + 0.01)
    env.last_renewed = 0.0

    # Renew attempt fails → the SAME acquire discards the dead container and
    # returns a fresh one (no request runs on the expired sandbox).
    second = await manager.acquire("s1")
    await _sleep_tasks(manager)

    assert second is not env
    assert len(provider.created) == 2
    assert provider.killed == ["sandbox-1"]


async def test_stale_env_discarded_and_never_restored() -> None:
    """A stale env is killed and replaced by a fresh container — restore must
    never be attempted for it."""

    class RestoringProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.restored: list[str] = []

        async def restore_session(
            self, session_id, timeout=None, bundle_specs=None, *, fingerprint=None
        ):
            self.restored.append(session_id)
            return FakeSandbox("restored-sandbox")

    provider = RestoringProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
    )

    env = await manager.acquire("s1")
    assert provider.restored == ["s1"]

    # Mark stale (backend container died) and acquire again.
    env._stale = True
    fresh = await manager.acquire("s1")
    await _sleep_tasks(manager)

    # Restore was NOT attempted for the stale env; fresh container created.
    assert provider.restored == ["s1"]  # only the first acquire restored
    assert provider.created == ["sandbox-1"]
    assert provider.killed == ["restored-sandbox"]
    assert fresh.sandbox.sandbox_id == "sandbox-1"


# ---------------------------------------------------------------------------
# Reaper — idle cleanup only, no renewal
# ---------------------------------------------------------------------------


async def test_reaper_kills_idle_sandboxes() -> None:
    manager = _manager()

    env = await manager.acquire("s1")
    # Age the env past the TTL (1h) so the reaper considers it idle.
    env.last_accessed = 0.0
    await manager._reap_once()
    await _sleep_tasks(manager)

    assert manager._provider.killed == ["sandbox-1"]
    assert manager._active == {}


async def test_reaper_does_not_renew_live_sandboxes() -> None:
    """The reaper only kills idle envs — renewal is on-acquire."""
    manager = _manager()

    await manager.acquire("s1")
    await manager._reap_once()  # env is live — reaper must NOT renew it

    assert manager._provider.renewed == []


# ---------------------------------------------------------------------------
# Session restore
# ---------------------------------------------------------------------------


async def test_acquire_restores_when_provider_has_sandbox() -> None:
    """A provider that can restore is used instead of creating fresh."""

    class RestoringProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.restored: list[str] = []
            self._existing = FakeSandbox("restored-sandbox")

        async def restore_session(
            self, session_id, timeout=None, bundle_specs=None, *, fingerprint=None
        ):
            self.restored.append(session_id)
            return self._existing

    provider = RestoringProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
    )

    env = await manager.acquire("s1")

    assert provider.restored == ["s1"]
    assert provider.created == []
    assert env.sandbox is provider._existing


async def test_mount_change_forces_fresh_create_even_with_restore() -> None:
    """Mount changes must not restore the old (killed) sandbox."""

    class RestoringProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.restored: list[str] = []

        async def restore_session(
            self, session_id, timeout=None, bundle_specs=None, *, fingerprint=None
        ):
            self.restored.append(session_id)
            return FakeSandbox("restored-sandbox")

    provider = RestoringProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
    )

    await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    await _sleep_tasks(manager)
    second = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    await _sleep_tasks(manager)

    # First acquire restored; mount change forced a fresh create (no restore).
    assert provider.restored == ["s1"]
    assert provider.created == ["sandbox-1"]
    assert provider.killed == ["restored-sandbox"]
    assert second.sandbox.sandbox_id == "sandbox-1"


# ---------------------------------------------------------------------------
# Multi-pod / multi-worker safety
# ---------------------------------------------------------------------------


class SharedProvider(FakeProvider):
    """Provider with a shared registry so two managers (pods) see each other's
    containers — simulating the OpenSandbox metadata discovery."""

    def __init__(self) -> None:
        super().__init__()
        self._registry: dict[str, FakeSandbox] = {}

    async def create_session(
        self,
        user_id=None,
        timeout=None,
        bundle_specs=None,
        *,
        session_id=None,
        volumes=None,
        env=None,
        fingerprint=None,
    ) -> FakeSandbox:
        sandbox = await super().create_session(
            session_id=session_id, env=env, fingerprint=fingerprint
        )
        if session_id:
            self._registry[session_id] = sandbox
        return sandbox

    async def restore_session(
        self, session_id, timeout=None, bundle_specs=None, *, fingerprint=None
    ):
        return self._registry.get(session_id)


async def test_distributed_lock_prevents_double_create() -> None:
    """Two managers (different instance ids) racing the same session must end
    with ONE container: the second restores the first's."""
    from private_gpt.components.environment.distributed import DistributedCoordinator

    provider = SharedProvider()
    m1 = _manager(
        sandbox_provider=provider,
        coordinator=DistributedCoordinator(instance_id="pod-a"),
    )
    m2 = _manager(
        sandbox_provider=provider,
        coordinator=DistributedCoordinator(instance_id="pod-b"),
    )

    env1, env2 = await asyncio.gather(m1.acquire("s1"), m2.acquire("s1"))

    assert len(provider.created) == 1  # no double-create leak
    assert env1.sandbox is env2.sandbox  # both managers ended on one container


async def test_reaper_does_not_kill_when_shared_activity_recent() -> None:
    """A reaper on pod A must not kill a sandbox pod B is actively using."""
    from private_gpt.components.environment.distributed import DistributedCoordinator

    coordinator = DistributedCoordinator(instance_id="pod-a")
    manager = _manager(coordinator=coordinator)

    env = await manager.acquire("s1")
    # Locally idle past TTL (this pod stopped using it)...
    env.last_accessed = 0.0
    # ...but pod B touched the shared clock recently.
    await coordinator.set_activity("s1")

    await manager._reap_once()
    await _sleep_tasks(manager)

    assert manager._provider.killed == []
    assert "s1" in manager._active


async def test_reaper_kills_when_shared_activity_old() -> None:
    from private_gpt.components.environment.distributed import DistributedCoordinator

    coordinator = DistributedCoordinator(instance_id="pod-a")
    manager = _manager(coordinator=coordinator)

    env = await manager.acquire("s1")
    env.last_accessed = 0.0
    # Shared clock also idle for longer than the TTL (60 min).
    async with _dist._fallback_guard:
        _dist._fallback_activity["sandbox:activity:s1"] = time.time() - 2 * 3600

    await manager._reap_once()
    await _sleep_tasks(manager)

    assert manager._provider.killed == ["sandbox-1"]
    assert manager._active == {}


async def test_fingerprint_passed_to_create_and_restore() -> None:
    """The manager computes a stable fingerprint and forwards it to both
    create_session and restore_session."""
    provider = FakeProvider()
    manager = _manager(sandbox_provider=provider)

    await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/")],
        sandbox_env={"TOKEN": "x"},
    )
    await _sleep_tasks(manager)

    assert provider.fingerprints == [
        manager._fingerprint([_bundle_mount("/mnt/skills/a/")], {"TOKEN": "x"})
    ]
