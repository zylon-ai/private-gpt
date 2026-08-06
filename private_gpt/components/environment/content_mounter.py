from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from private_gpt.components.sandbox.base import SandboxSession
    from private_gpt.components.sandbox.mount import MountSpec


def _volume_name(target: str, prefix: str = "bundle") -> str:
    return f"{prefix}-{hashlib.sha1(target.encode()).hexdigest()[:8]}"


class ContentMounter(ABC):
    """Knows how to get a storage-backed MountSpec's content into a sandbox.

    Implementations are composed in a prioritized list; the manager picks the
    first one whose can_handle() returns True for a given mount. Volume
    mounters declare their bind-mount before container creation via
    prepare_volume() and are no-ops in materialize(); copy-based mounters
    leave prepare_volume() returning None and do the work in materialize(),
    which is called lazily before the first exec() after the mount is
    registered.
    """

    @abstractmethod
    def can_handle(self, descriptor: MountSpec) -> bool:
        """Return True if this mounter can resolve this storage-backed mount."""

    async def prepare_volume(
        self, descriptor: MountSpec, session_id: str
    ) -> MountSpec | None:
        """Return a resolved MountSpec (source set) to bind at container creation.

        When non-None, the spec is wired into sandbox creation and materialize()
        must be a no-op for this descriptor. Default: None (use materialize()).
        """
        return None

    @abstractmethod
    async def materialize(
        self, descriptor: MountSpec, sandbox: SandboxSession
    ) -> None:
        """Write the content into the live sandbox at descriptor.target.

        Called lazily just before the first exec() after the mount is
        registered. Always overwrites — no path_exists() check needed;
        idempotency is by design.
        """


class FetchContentMounter(ContentMounter):
    """Materializes storage-backed mounts by calling their fetch() callable.

    Works with any storage backend — fetch() is injected at mount construction
    time by the SkillLoader (or equivalent). Writes to whatever filesystem the
    sandbox has, whether ephemeral or S3-backed.
    """

    def can_handle(self, descriptor: MountSpec) -> bool:
        return descriptor.storage is not None

    async def materialize(
        self, descriptor: MountSpec, sandbox: SandboxSession
    ) -> None:
        if descriptor.storage is not None:
            files = await descriptor.storage.fetch()
            await sandbox.initialize_mount(descriptor.target, files)


class LocalStorageContentMounter(ContentMounter):
    """Volume-mounts storage-backed mounts from a local storage root on the host.

    When skills are stored locally (storage_provider='local'), the bundle files
    already exist at storage_root/storage_prefix and are bind-mounted directly.
    When storage_provider='s3', the local directory may be empty; in that case
    prepare_volume() fetches from the storage backend and caches the files
    locally before returning the MountSpec — keeps the bind-mount the only
    write path so callers never need to write to read-only container paths.
    For mounts added lazily after container creation, materialize() re-fetches
    and writes directly into the sandbox (bind-mounts cannot be added post-start).
    """

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root

    def can_handle(self, descriptor: MountSpec) -> bool:
        return descriptor.storage is not None

    async def prepare_volume(
        self, descriptor: MountSpec, session_id: str
    ) -> MountSpec | None:
        from private_gpt.components.sandbox.mount import MountSpec

        if descriptor.storage is None:
            return None

        host_path = self._root / descriptor.storage.prefix
        if not host_path.is_dir() or not any(host_path.iterdir()):
            # Local directory is absent or empty — fetch from the storage backend
            # (S3 etc.) and cache locally so the bind-mount has content.
            files = await descriptor.storage.fetch()
            if not files:
                return None
            for f in files:
                dest = host_path / f.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.content)

        return MountSpec(
            name=_volume_name(descriptor.target, "stored"),
            target=descriptor.target,
            access=descriptor.access,
            source=host_path,
        )

    async def materialize(
        self, descriptor: MountSpec, sandbox: SandboxSession
    ) -> None:
        from private_gpt.components.sandbox.local import BashExecutorSandbox

        if descriptor.storage is None:
            return

        if isinstance(sandbox, BashExecutorSandbox):
            # Ensure skill files exist locally (fetch from backend if absent),
            # then register the host-path mapping in the path translator so
            # that subsequent exec() calls can resolve the canonical path.
            host_path = self._root / descriptor.storage.prefix
            if not host_path.is_dir() or not any(host_path.iterdir()):
                files = await descriptor.storage.fetch()
                if not files:
                    return
                for f in files:
                    dest = host_path / f.path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.content)
            sandbox.add_local_mount(
                descriptor.target, host_path, writable=descriptor.writable
            )
        else:
            files = await descriptor.storage.fetch()
            if files:
                await sandbox.initialize_mount(descriptor.target, files)
