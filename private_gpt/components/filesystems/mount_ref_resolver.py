"""Resolves Backend mount entries into directory-level MountSpec objects (T4.2).

The Backend emits generic mount entries (namespace + scope + path + target).
This component resolves each entry via the PathResolver and produces a
MountSpec that the sandbox can bind-mount.

A mount is always a directory — never a single file. File-level entries are
normalized to the containing directory (the resolved host dir when the path
is a directory, or its parent when the path is a file), so the agent works
over mount paths only.

Unresolvable or unauthorised entries are skipped (not fatal) — the
conversation continues without that file rather than failing entirely.
"""

from __future__ import annotations

import logging
import posixpath

from injector import inject, singleton

from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.components.sandbox.mount import MountSpec

logger = logging.getLogger(__name__)


@singleton
class MountRefResolver:
    """Resolves mount entries from the Backend mount plan into MountSpecs.

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

    def resolve(self, entries: list[MountEntry]) -> list[MountSpec]:
        """Convert a list of mount entries to bind-mountable directory MountSpecs.

        Silently skips unresolvable entries.
        """
        if not entries:
            return []

        mounts: list[MountSpec] = []
        for entry in entries:
            mount = self._resolve_one(entry)
            if mount is not None:
                mounts.append(mount)
        return mounts

    def _resolve_one(self, entry: MountEntry) -> MountSpec | None:
        try:
            # Verify namespace is known before attempting resolution
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

        if not host_path.exists():
            logger.debug(
                "Skipping mount entry (host path does not exist): %s -> %s",
                host_path,
                entry.target,
            )
            return None

        # Mounts are directories — never single files. If the resolved host
        # path is a file, mount its parent directory instead.
        source = host_path if host_path.is_dir() else host_path.parent
        target = _directory_target(entry.target, file_target=not host_path.is_dir())

        return MountSpec(
            name=f"mount-{entry.namespace}-{entry.artifact_id or entry.path[:16]}",
            target=target,
            access="ro" if entry.mode == "ro" else "rw",
            source=source,
            etag=entry.etag,
        )


def _directory_target(target: str, *, file_target: bool) -> str:
    """Normalize a Backend target to a directory mount path (ends with "/")."""
    if target.endswith("/"):
        return target
    if file_target:
        parent = posixpath.dirname(target.rstrip("/"))
        return parent.rstrip("/") + "/"
    return target.rstrip("/") + "/"
