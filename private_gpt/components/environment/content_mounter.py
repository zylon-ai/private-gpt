from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from private_gpt.components.sandbox.base import SandboxSession
    from private_gpt.components.sandbox.mount import Mount


def _volume_name(target: str, prefix: str = "mount") -> str:
    return f"{prefix}-{hashlib.sha1(target.encode()).hexdigest()[:8]}"


class ContentMounter(ABC):
    """Knows how to get a URI-backed Mount's content into a sandbox.

    Implementations are composed in a prioritized list; the manager picks the
    first one whose can_handle() returns True for a given mount. Volume
    mounters declare their bind-mount before container creation via
    prepare_volume() and are no-ops in materialize(); copy-based mounters
    leave prepare_volume() returning None and do the work in materialize(),
    which is called lazily before the first exec() after the mount is
    registered.
    """

    @abstractmethod
    def can_handle(self, descriptor: Mount) -> bool:
        """Return True if this mounter can resolve this URI-backed mount."""

    async def prepare_volume(
        self, descriptor: Mount, session_id: str
    ) -> Mount | None:
        """Return a resolved Mount (host_path set) to bind at container creation.

        When non-None, the spec is wired into sandbox creation and materialize()
        must be a no-op for this descriptor. Default: None (use materialize()).
        """
        return None

    @abstractmethod
    async def materialize(
        self, descriptor: Mount, sandbox: SandboxSession
    ) -> None:
        """Write the content into the live sandbox at descriptor.target.

        Called lazily just before the first exec() after the mount is
        registered. Always overwrites — no path_exists() check needed;
        idempotency is by design.
        """


class FetchContentMounter(ContentMounter):
    """Materializes URI-backed mounts by calling their fetch() callable.

    Works with any storage backend — fetch() is built from the mount's URI
    (see UriSource.from_uri) and writes to whatever filesystem the sandbox
    has, whether ephemeral or S3-backed.
    """

    def can_handle(self, descriptor: Mount) -> bool:
        return descriptor.uri_source is not None

    async def materialize(
        self, descriptor: Mount, sandbox: SandboxSession
    ) -> None:
        if descriptor.uri_source is not None:
            files = await descriptor.uri_source.fetch()
            await sandbox.initialize_mount(descriptor.target, files)


class LocalStorageContentMounter(ContentMounter):
    """Volume-mounts URI-backed mounts from a local storage root on the host.

    When skills are stored locally (storage_provider='local'), the files
    already exist under the storage root and are bind-mounted directly.
    When storage_provider='s3', the local directory may be empty; in that
    case prepare_volume() fetches from the URI and caches the files locally
    before returning the Mount — keeps the bind-mount the only write path so
    callers never need to write to read-only container paths. For mounts
    added lazily after container creation, materialize() re-fetches and
    writes directly into the sandbox (bind-mounts cannot be added post-start).
    """

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root

    def can_handle(self, descriptor: Mount) -> bool:
        return descriptor.uri_source is not None

    async def prepare_volume(
        self, descriptor: Mount, session_id: str
    ) -> Mount | None:
        from private_gpt.components.sandbox.mount import Mount

        if descriptor.uri_source is None:
            return None

        host_path = self._root / descriptor.uri_source.uri
        if not host_path.is_dir() or not any(host_path.iterdir()):
            # Local directory is absent or empty — fetch from the URI and cache
            # locally so the bind-mount has content.
            files = await descriptor.uri_source.fetch()
            if not files:
                return None
            for f in files:
                dest = host_path / f.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.content)
                dest.chmod(f.permissions)

        return Mount(
            name=_volume_name(descriptor.target, "stored"),
            target=descriptor.target,
            access=descriptor.access,
            host_path=host_path,
            etag=descriptor.etag,
        )

    async def materialize(
        self, descriptor: Mount, sandbox: SandboxSession
    ) -> None:
        from private_gpt.components.sandbox.local import BashExecutorSandbox

        if descriptor.uri_source is None:
            return

        if isinstance(sandbox, BashExecutorSandbox):
            # Ensure files exist locally (fetch from URI if absent), then
            # register the host-path mapping in the path translator so that
            # subsequent exec() calls can resolve the canonical path.
            host_path = self._root / descriptor.uri_source.uri
            if not host_path.is_dir() or not any(host_path.iterdir()):
                files = await descriptor.uri_source.fetch()
                if not files:
                    return
                for f in files:
                    dest = host_path / f.path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.content)
                    dest.chmod(f.permissions)
            sandbox.add_local_mount(
                descriptor.target, host_path, writable=descriptor.writable
            )
        else:
            files = await descriptor.uri_source.fetch()
            if files:
                await sandbox.initialize_mount(descriptor.target, files)
