"""Tests for HydratingEnvironmentManager (dev-only etag-ledger hydration)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from private_gpt.components.environment.hydration import HydratingEnvironmentManager
from private_gpt.components.sandbox.mount import (
    Mount,
    MountFile,
    MountSource,
    UriSource,
)
from private_gpt.settings.settings import NamespaceConfig

if TYPE_CHECKING:
    from pathlib import Path


class FakeDelegate:
    def __init__(self) -> None:
        self.acquired: list[tuple[str | None, list[Mount] | None]] = []
        self.released: list[str] = []

    async def acquire(self, session_id, mounts=None, sandbox_env=None):
        self.acquired.append((session_id, mounts))
        return "env"

    def release(self, session_id: str) -> None:
        self.released.append(session_id)


def _file_mount(
    *,
    uri: str,
    etag: str | None,
    host_path: Path,
    target: str = "/home/agent/workspace/potato.md",
    fetch=None,
) -> Mount:
    async def _default_fetch() -> list[MountFile]:
        return [MountFile(path="potato.md", content=b"# Potato")]

    return Mount(
        target=target,
        access="rw",
        host_path=host_path,
        uri_source=UriSource(
            uri=uri,
            fetch=fetch or _default_fetch,
        ),
        source=MountSource(
            namespace="artifacts",
            scope="thread-1",
            path="art-1/_content.md",
        ),
        etag=etag,
    )


def _manager(root: Path, hydration: bool = True) -> HydratingEnvironmentManager:
    cfg = NamespaceConfig(root=str(root), default_mode="rw", hydration=hydration)
    return HydratingEnvironmentManager(
        manager=FakeDelegate(), namespaces={"artifacts": cfg}
    )


async def test_hydrates_missing_file_and_writes_ledger(tmp_path: Path) -> None:
    host = tmp_path / "volumes" / "artifacts" / "thread-1" / "art-1" / "_content.md"
    manager = _manager(tmp_path / "volumes" / "artifacts")

    mounts = [_file_mount(uri="s3://bucket/_content.md", etag="v1", host_path=host)]
    env = await manager.acquire("s1", mounts=mounts)

    assert env == "env"
    assert host.read_bytes() == b"# Potato"
    ledger = (
        tmp_path
        / "volumes"
        / "artifacts"
        / ".hydration"
        / "thread-1/art-1/_content.md.json"
    )
    assert json.loads(ledger.read_text()) == {"etag": "v1"}
    # The mount was passed through to the delegate unchanged.
    assert manager._manager.acquired[0][1] == mounts


async def test_skips_when_etag_unchanged(tmp_path: Path) -> None:
    host = tmp_path / "artifacts" / "thread-1" / "art-1" / "_content.md"
    host.parent.mkdir(parents=True)
    host.write_bytes(b"original")

    calls: list[str] = []

    async def fetch() -> list[MountFile]:
        calls.append("fetch")
        return [MountFile(path="potato.md", content=b"# Potato")]

    manager = _manager(tmp_path / "artifacts")
    mount = _file_mount(
        uri="s3://bucket/_content.md", etag="v1", host_path=host, fetch=fetch
    )
    await manager.acquire("s1", mounts=[mount])
    await manager.acquire("s1", mounts=[mount])

    # Fetched only once: the second acquire saw a fresh ledger and skipped.
    assert calls == ["fetch"]
    assert host.read_bytes() == b"# Potato"


async def test_refetches_when_etag_changes(tmp_path: Path) -> None:
    host = tmp_path / "artifacts" / "thread-1" / "art-1" / "_content.md"
    host.parent.mkdir(parents=True)
    host.write_bytes(b"old")

    calls: list[str] = []

    async def fetch() -> list[MountFile]:
        calls.append("fetch")
        return [MountFile(path="_content.md", content=b"new")]

    manager = HydratingEnvironmentManager(
        manager=FakeDelegate(),
        namespaces={
            "artifacts": NamespaceConfig(
                root=str(tmp_path / "artifacts"), hydration=True
            )
        },
    )
    mount = _file_mount(uri="s3://bucket/_content.md", etag="v1", host_path=host)
    mount.uri_source = UriSource(uri="s3://bucket/_content.md", fetch=fetch)
    await manager.acquire("s1", mounts=[mount])

    mount.etag = "v2"
    await manager.acquire("s1", mounts=[mount])

    assert calls == ["fetch", "fetch"]
    assert host.read_bytes() == b"new"


async def test_drops_mount_when_fetch_empty(tmp_path: Path) -> None:
    host = tmp_path / "artifacts" / "thread-1" / "art-1" / "_content.md"
    manager = HydratingEnvironmentManager(
        manager=FakeDelegate(),
        namespaces={
            "artifacts": NamespaceConfig(
                root=str(tmp_path / "artifacts"), hydration=True
            )
        },
    )

    async def _empty() -> list[MountFile]:
        return []

    mount = _file_mount(
        uri="s3://bucket/_content.md",
        etag="v1",
        host_path=host,
        fetch=_empty,
    )
    await manager.acquire("s1", mounts=[mount])

    assert manager._manager.acquired[0][1] == []
