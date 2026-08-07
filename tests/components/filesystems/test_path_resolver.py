"""Tests for the filesystem path resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from private_gpt.components.filesystems.path_resolver import (
    InvalidPathError,
    PathEscapeError,
    PathResolver,
)
from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig

if TYPE_CHECKING:
    from pathlib import Path


def _make_resolver(tmp_path: Path, namespaces: dict[str, str]) -> PathResolver:
    """Build a PathResolver with real roots created under tmp_path."""
    from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry

    ns_configs: dict[str, NamespaceConfig] = {}
    for name, subdir in namespaces.items():
        root = tmp_path / subdir
        root.mkdir(parents=True, exist_ok=True)
        ns_configs[name] = NamespaceConfig(root=str(root), default_mode="rw")

    fs_settings = FilesystemsSettings(namespaces=ns_configs)

    class _FakeSettings:
        filesystems = fs_settings

    registry = NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]
    return PathResolver(registry=registry)


class TestValidPaths:
    def test_simple_path(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        result = resolver.resolve("session", "scope-1", "outputs/result.png")
        assert str(result).endswith("scope-1/outputs/result.png")

    def test_empty_scope(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        result = resolver.resolve("session", "", "file.txt")
        assert "file.txt" in str(result)

    def test_empty_path(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        result = resolver.resolve("session", "scope-1", "")
        root = tmp_path / "vols" / "scope-1"
        assert str(result).startswith(str(root))

    def test_nested_path(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"artifacts": "arts"})
        result = resolver.resolve("artifacts", "org-1", "project-1/art-1")
        assert "org-1/project-1/art-1" in str(result)


class TestTraversalRejection:
    def test_dotdot_in_path_rejected(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(InvalidPathError, match=r"\.\."):
            resolver.resolve("session", "scope-1", "../other")

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(InvalidPathError, match="relative"):
            resolver.resolve("session", "scope-1", "/etc/passwd")

    def test_scope_with_slash_rejected(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(InvalidPathError, match="single path segment"):
            resolver.resolve("session", "a/b", "file.txt")

    def test_scope_with_dotdot_rejected(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(InvalidPathError, match="single path segment"):
            resolver.resolve("session", "..", "file.txt")

    def test_dotdot_in_middle_of_path(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(InvalidPathError):
            resolver.resolve("session", "scope-1", "a/../../etc/passwd")


class TestSymlinkEscape:
    def test_symlink_escaping_root_rejected(self, tmp_path: Path) -> None:
        """A symlink inside the namespace pointing outside the root must be rejected."""
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        root = tmp_path / "vols"

        # Create a symlink inside the namespace root pointing outside it
        evil_dir = tmp_path / "secret"
        evil_dir.mkdir()
        link = root / "escape_link"
        link.symlink_to(evil_dir)

        with pytest.raises(PathEscapeError, match="outside namespace root"):
            resolver.resolve("session", "", "escape_link")


class TestUnknownNamespace:
    def test_unknown_namespace_raises_key_error(self, tmp_path: Path) -> None:
        resolver = _make_resolver(tmp_path, {"session": "vols"})
        with pytest.raises(KeyError, match="unknown"):
            resolver.resolve("unknown", "scope-1", "file.txt")
