"""Resolves Backend mount entries into local VolumeSpec objects (T4.2).

The Backend emits generic mount entries (namespace + scope + path + target).
This component resolves each entry via the PathResolver and produces a
VolumeSpec that the sandbox can bind-mount.

Unresolvable or unauthorised entries are skipped (not fatal) — the
conversation continues without that file rather than failing entirely.
"""

from __future__ import annotations

import logging

from injector import inject, singleton

from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.components.sandbox.mount import VolumeSpec

logger = logging.getLogger(__name__)


@singleton
class MountRefResolver:
    """Resolves mount entries from the Backend mount plan into VolumeSpecs.

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

    def resolve(self, entries: list[MountEntry]) -> list[VolumeSpec]:
        """Convert a list of mount entries to bind-mountable VolumeSpecs.

        Silently skips unresolvable entries.
        """
        if not entries:
            return []

        volumes: list[VolumeSpec] = []
        for entry in entries:
            volume = self._resolve_one(entry)
            if volume is not None:
                volumes.append(volume)
        return volumes

    def _resolve_one(self, entry: MountEntry) -> VolumeSpec | None:
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

        return VolumeSpec(
            name=f"mount-{entry.namespace}-{entry.artifact_id or entry.path[:16]}",
            host_path=host_path,
            mount_path=entry.target,
            read_only=(entry.mode == "ro"),
        )
