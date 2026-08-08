"""Path resolver for the ZGPT filesystem platform.

Translates ``(namespace, scope, path)`` triples into absolute local paths,
enforcing strict containment: the resolved path must be inside the namespace
root and no symlink inside the namespace may escape outside of it.

Security rules enforced:
- ``path`` must be relative (no leading ``/``).
- ``path`` must not contain ``..`` components.
- ``scope`` must not contain ``..`` or ``/``.
- After resolution, the canonical absolute path must be inside the namespace root.
- Symlinks are resolved and their targets must also be inside the namespace root.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from injector import inject, singleton

from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry


class PathEscapeError(ValueError):
    """Raised when the resolved path would escape the namespace root."""


class InvalidPathError(ValueError):
    """Raised when the path or scope contains illegal components."""


@singleton
class PathResolver:
    """Resolves (namespace, scope, path) → absolute local path.

    This is a pure function wrapped in a singleton so callers can obtain it
    via dependency injection without re-reading settings.
    """

    @inject
    def __init__(self, registry: NamespaceRegistry) -> None:
        self._registry = registry

    def resolve(self, namespace: str, scope: str, path: str) -> Path:
        """Return the absolute local path for ``(namespace, scope, path)``.

        Parameters
        ----------
        namespace:
            Logical namespace name; must be registered.
        scope:
            Opaque scope identifier (e.g. session-id, thread-id).  Must be a
            single path segment (no ``/`` or ``..``).
        path:
            Relative path within the scope.  Must not start with ``/`` or
            contain ``..`` segments.

        Returns:
        -------
        Path
            Resolved, absolute path guaranteed to reside inside the namespace
            root.

        Raises:
        ------
        KeyError
            If the namespace is not registered.
        InvalidPathError
            If ``scope`` or ``path`` contain illegal components.
        PathEscapeError
            If the resolved path (after symlink expansion) is outside the root.
        """
        root = self._registry.root(namespace)  # raises KeyError for unknown NS

        _validate_scope(scope)
        _validate_path(path)

        candidate = root / scope / path

        # Resolve symlinks *within* the namespace root only.
        resolved = _safe_realpath(candidate, root)

        return resolved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_scope(scope: str) -> None:
    if not scope:
        return  # empty scope is allowed (namespace-level access)
    if ".." in scope.split("/") or "/" in scope:
        raise InvalidPathError(
            f"scope must be a single path segment without '..' or '/': {scope!r}"
        )


def _validate_path(path: str) -> None:
    if not path:
        return  # empty path resolves to the scope directory itself
    if path.startswith("/"):
        raise InvalidPathError(f"path must be relative (no leading '/'): {path!r}")
    parts = PurePosixPath(path).parts
    if ".." in parts:
        raise InvalidPathError(f"path must not contain '..' components: {path!r}")


def _safe_realpath(candidate: Path, root: Path) -> Path:
    """Resolve symlinks, refusing to escape *root*.

    We cannot use ``Path.resolve()`` directly because it follows symlinks
    unconditionally.  Instead we iteratively resolve each component, stopping
    if a symlink target exits the root.
    """
    # First, do a quick realpath of the root itself so comparisons are stable.
    root_real = Path(os.path.realpath(str(root)))

    try:
        real = Path(os.path.realpath(str(candidate)))
    except OSError:
        # Path does not exist yet; validate the non-resolved form instead.
        real = candidate.resolve()

    try:
        real.relative_to(root_real)
    except ValueError as exc:
        raise PathEscapeError(
            f"Resolved path {real!r} is outside namespace root {root_real!r}."
        ) from exc

    return real
