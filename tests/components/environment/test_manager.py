"""Tests for EnvironmentManager mount-change recreation and eager renewal."""

from __future__ import annotations

from pathlib import Path

from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.sandbox.mount import Mount, UriSource

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
        self._counter = 0

    async def restore_session(self, session_id, timeout=None, bundle_specs=None):
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
    ) -> FakeSandbox:
        self._counter += 1
        sandbox_id = f"sandbox-{self._counter}"
        self.created.append(sandbox_id)
        return FakeSandbox(sandbox_id)

    async def renew_session(self, session) -> None:
        self.renewed.append(session.sandbox_id)

    async def kill_session(self, session, session_id=None) -> None:
        session.killed = True
        self.killed.append(session.sandbox_id)


def _manager(**kwargs) -> EnvironmentManager:
    return EnvironmentManager(
        sandbox_provider=FakeProvider(),
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
# Mount-change recreation
# ---------------------------------------------------------------------------


async def test_acquire_recreates_when_bundles_change() -> None:
    manager = _manager(recreate_on_mount_change=True)

    first = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    await _sleep_tasks(manager)
    second = await manager.acquire(
        "s1",
        mounts=[_bundle_mount("/mnt/skills/a/"), _bundle_mount("/mnt/skills/b/")],
    )
    await _sleep_tasks(manager)

    assert first is not second
    assert len(manager._provider.created) == 2
    # The old sandbox was killed, not materialized into.
    assert manager._provider.killed == ["sandbox-1"]
    assert len(manager._provider.renewed) == 0


async def test_acquire_reuses_when_mounts_unchanged() -> None:
    manager = _manager(recreate_on_mount_change=True)

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
    manager = _manager(recreate_on_mount_change=True)

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
    manager = _manager(recreate_on_mount_change=True)

    first = await manager.acquire("s1", sandbox_env={"TOKEN": "a"})
    await _sleep_tasks(manager)
    second = await manager.acquire("s1", sandbox_env={"TOKEN": "b"})
    await _sleep_tasks(manager)

    assert first is not second
    assert len(manager._provider.created) == 2
    assert manager._provider.killed == ["sandbox-1"]


async def test_acquire_reuses_when_mounts_unchanged_again() -> None:
    manager = _manager(recreate_on_mount_change=True)

    first = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    second = await manager.acquire("s1", mounts=[_bundle_mount("/mnt/skills/a/")])
    await _sleep_tasks(manager)

    assert first is second
    assert len(manager._provider.created) == 1


async def test_acquire_recreates_when_bundle_removed() -> None:
    manager = _manager(recreate_on_mount_change=True)

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


async def test_acquire_recreates_when_eager_renew_fails() -> None:
    class FlakyProvider(FakeProvider):
        async def renew_session(self, session) -> None:
            raise RuntimeError("sandbox expired")

    provider = FlakyProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,
        layout_mounter=FakeLayout(),
        content_mounters=[],
        ttl_seconds=3600,
        renew_on_acquire=True,
        recreate_on_mount_change=True,
    )

    first = await manager.acquire("s1")
    await _sleep_tasks(manager)
    second = await manager.acquire("s1")  # eager renew fails → recreate
    await _sleep_tasks(manager)

    assert first is not second
    assert len(provider.created) == 2
    assert provider.killed == ["sandbox-1"]


# ---------------------------------------------------------------------------
# Eager renewal on acquire + reaper behavior
# ---------------------------------------------------------------------------


async def test_renew_on_acquire_renews_eagerly() -> None:
    manager = _manager(renew_on_acquire=True)

    await manager.acquire("s1")
    await manager.acquire("s1")  # reuse → eager renew

    assert manager._provider.renewed == ["sandbox-1"]


async def test_renew_on_acquire_disables_reaper_renewal() -> None:
    manager = _manager(renew_on_acquire=True)

    await manager.acquire("s1")
    await manager._reap_once()  # env is live, must NOT be renewed by the reaper

    assert manager._provider.renewed == []


async def test_reaper_still_renews_when_not_eager() -> None:
    manager = _manager(renew_on_acquire=False)

    await manager.acquire("s1")
    await manager._reap_once()  # env is live → reaper renews it

    assert manager._provider.renewed == ["sandbox-1"]


async def test_reaper_kills_idle_sandboxes() -> None:
    manager = _manager(renew_on_acquire=True, recreate_on_mount_change=True)

    env = await manager.acquire("s1")
    # Age the env past the TTL (1h) so the reaper considers it idle.
    env.last_accessed = 0.0
    await manager._reap_once()
    await _sleep_tasks(manager)

    assert manager._provider.killed == ["sandbox-1"]
    assert manager._active == {}
