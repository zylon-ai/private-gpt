"""Resolves Backend mount entries into directory-level Mount objects.

The Backend emits generic mount entries (namespace + scope + path + target,
or a content ``uri``). This component resolves each entry into a Mount the
sandbox can bind-mount.

A mount is always a directory — never a single file. File-level entries are
normalized to the containing directory (the resolved host dir when the path
is a directory, or its parent when the path is a file), so the agent works
over mount paths only.

Unresolvable namespace references are skipped (not fatal). URI-backed
entries remain generic and retain their namespace identity for the selected
runtime provider to resolve.
"""

from __future__ import annotations

import logging
import posixpath
from collections import defaultdict
from dataclasses import dataclass

from injector import inject, singleton

from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.components.sandbox.mount import Mount, UriSource

logger = logging.getLogger(__name__)


@dataclass
class _ResolvedMount:
    entry: MountEntry
    mount: Mount


@singleton
class MountRefResolver:
    """Resolves mount entries from the Backend mount plan into Mounts.

    A ``uri`` entry becomes a lazy mount (content fetched from the URI only
    when the backing folder is empty). A namespace/scope/path entry is
    resolved to a local host folder (eager when it already has content).

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
        """Convert a list of mount entries to bind-mountable directory Mounts.

        Silently skips unresolvable entries.
        """
        if not entries:
            return []

        resolved: list[_ResolvedMount] = []
        for entry in entries:
            mount = self._resolve_one(entry)
            if mount is not None:
                resolved.append(mount)
        return _merge_compatible_mounts(resolved)

    def _resolve_one(self, entry: MountEntry) -> _ResolvedMount | None:
        target = _directory_target(entry.target, file_target=True)

        if entry.uri:
            return _ResolvedMount(
                entry=entry,
                mount=Mount(
                    name=f"mount-{entry.namespace}-{entry.scope or entry.path[:16]}",
                    target=target,
                    access="ro" if entry.mode == "ro" else "rw",
                    uri_source=UriSource.from_uri(
                        entry.uri,
                        filename=posixpath.basename(entry.target.rstrip("/")) or None,
                    ),
                    source_namespace=entry.namespace,
                    source_scope=entry.scope,
                    source_path=entry.path,
                    etag=entry.etag,
                ),
            )

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

        return _ResolvedMount(
            entry=entry,
            mount=Mount(
                name=f"mount-{entry.namespace}-{entry.scope or entry.path[:16]}",
                target=target,
                access="ro" if entry.mode == "ro" else "rw",
                host_path=source,
                source_namespace=entry.namespace,
                source_scope=entry.scope,
                source_path=entry.path,
                etag=entry.etag,
            ),
        )


def _directory_target(target: str, *, file_target: bool) -> str:
    """Normalize a Backend target to a directory mount path (ends with "/")."""
    if target.endswith("/"):
        return target
    if file_target:
        parent = posixpath.dirname(target.rstrip("/"))
        return parent.rstrip("/") + "/"
    return target.rstrip("/") + "/"


def _group_by_target(resolved: list[_ResolvedMount]) -> list[list[_ResolvedMount]]:
    groups: dict[str, list[_ResolvedMount]] = defaultdict(list)
    for item in resolved:
        groups[item.mount.target].append(item)
    return list(groups.values())


def _merge_compatible_mounts(resolved: list[_ResolvedMount]) -> list[Mount]:
    """Merge only mounts that can share one directory without losing data.

    URI-backed files targeting the same directory are a generic composite
    mount. Mixed host/URI mounts and overlapping directories are deliberately
    left separate so the runtime provider can validate and reject conflicts;
    this layer must not contain artifact-specific conflict policy.
    """
    merged: list[Mount] = []
    for group in _group_by_target(resolved):
        if len(group) == 1:
            merged.append(group[0].mount)
            continue

        mounts = [item.mount for item in group]
        if all(mount.uri_source is not None for mount in mounts):
            merged.append(_merge_mount_group(group))
        else:
            merged.extend(mounts)
    return merged


def _merge_mount_group(group: list[_ResolvedMount]) -> Mount:
    if len(group) == 1:
        return group[0].mount

    mounts = [item.mount for item in group]
    uri_sources = [mount.uri_source for mount in mounts]
    if all(source is not None for source in uri_sources):
        sources = [source for source in uri_sources if source is not None]
        first, *additional = sources
        additional_sources: list[tuple[str, str | None]] = []
        for source in additional:
            additional_sources.extend(source.sources)
        mounts[0].uri_source = UriSource.from_uri(
            first.uri,
            filename=first.filename,
            additional_sources=additional_sources,
        )
        mounts[0].access = (
            "rw" if any(mount.access == "rw" for mount in mounts) else "ro"
        )
        return mounts[0]

    raise ValueError("Only URI-backed mounts can be merged")
