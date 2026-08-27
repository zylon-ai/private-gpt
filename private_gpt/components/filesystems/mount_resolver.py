"""Resolves Backend mount entries into bind-mountable Mount objects.

Each ``MountEntry`` becomes exactly one :class:`Mount` with an exact target:

- a **folder** mount when the entry target ends with ``/``,
- a **file** mount otherwise (the target has a filename).

There is deliberately no target mangling, no parent-directory fallback and no
merging: one entry -> one bind volume, mirroring Docker ``-v`` semantics. The
runtime validates conflicts (exact-target collisions) and fails fast.

Host content is located in this order:

1. when the entry carries an ``s3://`` URI, the object key is resolved under
   the namespace root,
2. otherwise ``(namespace, scope, path)`` is resolved inside the namespace root.

Missing host paths are only kept when the namespace has hydration enabled so
the host file can be materialized from the URI before sandbox creation.
Otherwise a missing host path is skipped — bind-mounting a non-existent path
would create empty directories.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse

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
    from private_gpt.settings.settings import NamespaceConfig

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
            config = self._registry.get(entry.namespace)
        except KeyError:
            logger.debug(
                "Skipping mount entry (namespace='%s' not registered): %s",
                entry.namespace,
                entry.target,
            )
            return None

        try:
            host_path = self._host_path(entry, config)
        except (InvalidPathError, PathEscapeError, KeyError) as exc:
            logger.debug(
                "Skipping mount entry (resolution failed: %s): namespace=%s scope=%s path=%s uri=%s",
                exc,
                entry.namespace,
                entry.scope,
                entry.path,
                entry.uri,
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

        if not host_path.exists():
            # Only keep a missing host path when hydration can materialize it.
            if uri_source is None or not config.hydration:
                logger.debug(
                    "Skipping mount entry (host path missing): %s -> %s",
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

    def _host_path(self, entry: MountEntry, config: NamespaceConfig) -> Path:
        """Resolve the exact host file/folder for one mount entry."""
        root = Path(config.root)
        if entry.uri:
            from_uri = _host_path_from_s3_uri(root, entry.uri)
            if from_uri is not None:
                return from_uri
        return self._resolver.resolve(entry.namespace, entry.scope, entry.path)


def _host_path_from_s3_uri(root: Path, uri: str) -> Path | None:
    """Map ``s3://bucket/object/key`` onto ``{namespace_root}/object/key``.

    Durable content is stored under bucket-relative keys (for example
    ``{org}/{project}/{artifact}/_content.md``). When the namespace root is the
    mounted bucket (or a local mirror of it), the object key is the
    host-relative path.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        return None
    key = parsed.path.lstrip("/")
    if not key:
        return None
    parts = PurePosixPath(key).parts
    if any(part in ("", ".", "..") for part in parts):
        return None
    candidate = root.joinpath(*parts)
    root_resolved = root.resolve(strict=False)
    try:
        candidate.resolve(strict=False).relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


# Maximum volume-name length imposed by Kubernetes (DNS label, RFC 1123) and
# respected by Docker Compose / Swarm.  The fixed skeleton is
# ``mount-{ns}-{digest16}`` (23 chars overhead), leaving 40 chars for the
# namespace slug.
_VOLUME_NAME_MAX = 63
_VOLUME_NAME_OVERHEAD = len("mount-") + len("-") + 16  # 23
_VOLUME_NS_MAX = _VOLUME_NAME_MAX - _VOLUME_NAME_OVERHEAD  # 40


def _slugify_namespace(name: str) -> str:
    """Lower-case alphanumeric slug; runs of invalid chars become a single '-'."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return slug.strip("-")


def _volume_name(entry: MountEntry) -> str:
    """Deterministic, unique volume name for a mount entry.

    Volume names must be unique per sandbox and must not exceed 63 characters
    — the Kubernetes DNS-label limit, also enforced by Docker Compose / Swarm.

    Two entries in the same namespace/scope (e.g. two files of one thread)
    collide if the name only carries namespace+scope, so the exact target —
    which the runtime guarantees to be unique per mount — is hashed in.
    """
    digest = hashlib.sha1(entry.target.encode("utf-8")).hexdigest()[:16]
    ns_slug = _slugify_namespace(entry.namespace)[:_VOLUME_NS_MAX]
    return f"mount-{ns_slug}-{digest}"
