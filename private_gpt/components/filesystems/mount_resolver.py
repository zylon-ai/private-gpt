"""Resolves Backend mount entries into bind-mountable Mount objects.

Each ``MountEntry`` becomes exactly one :class:`Mount` with an exact target:

- a **folder** mount when the entry target ends with ``/``,
- a **file** mount otherwise (the target has a filename).

There is deliberately no target mangling, no parent-directory fallback and no
merging: one entry -> one bind volume, mirroring Docker ``-v`` semantics. The
runtime validates conflicts (exact-target collisions) and fails fast.

Content is located on the host by resolving ``(namespace, scope, path)`` to
the exact host file/folder inside the namespace root. When the host path does
not exist yet but the entry carries a ``uri``, the mount is still emitted and
the hydration layer (development only) fills the host path before the sandbox
is created. Entries that cannot be located at all are skipped — a missing
optional file must not abort the turn.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.components.sandbox.mount import Mount, MountSource, UriSource

if TYPE_CHECKING:
    from private_gpt.components.filesystems.mount_entry import MountEntry

logger = logging.getLogger(__name__)


@singleton
class MountResolver:
    """Resolves mount entries from the Backend mount plan into Mounts.

    Skips entries whose namespace is not registered or whose path is
    invalid/escaping — a missing optional file must not abort the turn.
    """

    @inject
    def __init__(
        self,
        registry: NamespaceRegistry,
        resolver: PathResolver,
    ) -> None:
        self._registry = registry
        self._resolver = resolver

    def resolve(self, entries: list[MountEntry]) -> list[Mount]:
        """Convert a list of mount entries to exact bind-mountable Mounts."""
        if not entries:
            return []

        resolved: list[Mount] = []
        for entry in entries:
            mount = self._resolve_one(entry)
            if mount is not None:
                resolved.append(mount)
        return resolved

    def _resolve_one(self, entry: MountEntry) -> Mount | None:
        try:
            # Verify namespace is known before attempting resolution.
            self._registry.get(entry.namespace)
        except KeyError:
            logger.debug(
                "Skipping mount entry (namespace='%s' not registered): %s",
                entry.namespace,
                entry.target,
            )
            return None

        try:
            host_path = self._resolver.resolve(entry.namespace, entry.scope, entry.path)
        except (InvalidPathError, PathEscapeError, KeyError) as exc:
            logger.debug(
                "Skipping mount entry (resolution failed: %s): namespace=%s scope=%s path=%s",
                exc,
                entry.namespace,
                entry.scope,
                entry.path,
            )
            return None

        uri_source = None
        if entry.uri:
            filename = (
                posixpath.basename(entry.target.rstrip("/"))
                if not entry.target.endswith("/")
                else None
            )
            uri_source = UriSource.from_uri(entry.uri, filename=filename)

        if not host_path.exists() and uri_source is None:
            # Nothing on the host and nothing to hydrate from: the content is
            # gone (or was never created). Skipping is safer than binding a
            # bogus empty path.
            logger.debug(
                "Skipping mount entry (no host content and no URI): %s -> %s",
                host_path,
                entry.target,
            )
            return None

        return Mount(
            name=_volume_name(entry),
            target=entry.target,
            access="ro" if entry.mode == "ro" else "rw",
            host_path=host_path,
            uri_source=uri_source,
            source=MountSource(
                namespace=entry.namespace,
                scope=entry.scope,
                path=entry.path,
            ),
            etag=entry.etag,
        )


def _volume_name(entry: MountEntry) -> str:
    """Deterministic, unique volume name for a mount entry.

    Volume names must be unique per sandbox (OpenSandbox rejects duplicates).
    Two entries in the same namespace/scope (e.g. two artifacts of one thread)
    collide if the name only carries namespace+scope, so the exact target —
    which the runtime guarantees to be unique per mount — is hashed in.
    """
    digest = hashlib.sha1(entry.target.encode("utf-8")).hexdigest()[:16]
    return f"mount-{entry.namespace}-{digest}"
