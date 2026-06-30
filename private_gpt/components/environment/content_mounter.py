from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from private_gpt.components.sandbox.base import SandboxSession
    from private_gpt.components.sandbox.content_bundle import ContentBundle
    from private_gpt.components.sandbox.mount import VolumeSpec


def _volume_name(canonical_path: str, prefix: str = "bundle") -> str:
    return f"{prefix}-{hashlib.sha1(canonical_path.encode()).hexdigest()[:8]}"


class ContentMounter(ABC):
    """Knows how to get a ContentBundle's content into a running sandbox.

    Implementations are composed in a prioritized list; the manager picks the
    first one whose can_handle() returns True for a given descriptor. Volume
    mounters declare their bind-mount before container creation via
    prepare_volume() and are no-ops in materialize(); copy-based mounters
    leave prepare_volume() returning None and do the work in materialize(),
    which is called lazily before the first exec() after the bundle is registered.
    """

    @abstractmethod
    def can_handle(self, descriptor: ContentBundle) -> bool:
        """Return True if this mounter can materialize this descriptor type."""

    async def prepare_volume(
        self, descriptor: ContentBundle, session_id: str
    ) -> VolumeSpec | None:
        """Return a VolumeSpec to bind-mount this content at container creation.

        When non-None, the spec is wired into sandbox creation and materialize()
        must be a no-op for this descriptor. Default: None (use materialize()).
        """
        return None

    @abstractmethod
    async def materialize(
        self, descriptor: ContentBundle, sandbox: SandboxSession
    ) -> None:
        """Write the content into the live sandbox at descriptor.canonical_path.

        Called lazily just before the first exec() after the bundle is registered.
        Always overwrites — no path_exists() check needed; idempotency is by design.
        """


class InlineContentMounter(ContentMounter):
    """Materializes ContentBundle instances whose files are already in memory."""

    def can_handle(self, descriptor: ContentBundle) -> bool:
        from private_gpt.components.sandbox.content_bundle import StoredBundle

        return not isinstance(descriptor, StoredBundle)

    async def materialize(
        self, descriptor: ContentBundle, sandbox: SandboxSession
    ) -> None:
        await sandbox.initialize_mount(descriptor.canonical_path, descriptor.files)


class FetchContentMounter(ContentMounter):
    """Materializes StoredBundle instances by calling their fetch() callable.

    Works with any storage backend — fetch() is injected at bundle construction
    time by the SkillLoader (or equivalent). Writes to whatever filesystem the
    sandbox has, whether ephemeral or S3-backed.
    """

    def can_handle(self, descriptor: ContentBundle) -> bool:
        from private_gpt.components.sandbox.content_bundle import StoredBundle

        return isinstance(descriptor, StoredBundle)

    async def materialize(
        self, descriptor: ContentBundle, sandbox: SandboxSession
    ) -> None:
        from private_gpt.components.sandbox.content_bundle import StoredBundle

        if isinstance(descriptor, StoredBundle):
            files = await descriptor.fetch()
            await sandbox.initialize_mount(descriptor.canonical_path, files)


class LocalStorageContentMounter(ContentMounter):
    """Volume-mounts StoredBundle instances from a local storage root on the host.

    When skills are stored locally (storage_provider='local'), the bundle files
    already exist at storage_root/storage_prefix and are bind-mounted directly.
    When storage_provider='s3', the local directory may be empty; in that case
    prepare_volume() fetches from the storage backend and caches the files
    locally before returning the VolumeSpec — keeps the bind-mount the only
    write path so callers never need to write to read-only container paths.
    For bundles added lazily after container creation, materialize() re-fetches
    and writes directly into the sandbox (bind-mounts cannot be added post-start).
    """

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root

    def can_handle(self, descriptor: ContentBundle) -> bool:
        from private_gpt.components.sandbox.content_bundle import StoredBundle

        return isinstance(descriptor, StoredBundle)

    async def prepare_volume(
        self, descriptor: ContentBundle, session_id: str
    ) -> VolumeSpec | None:
        from private_gpt.components.sandbox.content_bundle import StoredBundle
        from private_gpt.components.sandbox.mount import VolumeSpec

        if not isinstance(descriptor, StoredBundle):
            return None

        host_path = self._root / descriptor.storage_prefix
        if not host_path.is_dir() or not any(host_path.iterdir()):
            # Local directory is absent or empty — fetch from the storage backend
            # (S3 etc.) and cache locally so the bind-mount has content.
            files = await descriptor.fetch()
            if not files:
                return None
            for f in files:
                dest = host_path / f.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.content)

        return VolumeSpec(
            name=_volume_name(descriptor.canonical_path, "stored"),
            host_path=host_path,
            mount_path=descriptor.canonical_path,
            read_only=not descriptor.writable,
        )

    async def materialize(
        self, descriptor: ContentBundle, sandbox: SandboxSession
    ) -> None:
        from private_gpt.components.sandbox.content_bundle import StoredBundle

        if isinstance(descriptor, StoredBundle):
            files = await descriptor.fetch()
            if files:
                await sandbox.initialize_mount(descriptor.canonical_path, files)
