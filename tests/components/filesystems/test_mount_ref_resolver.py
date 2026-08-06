"""Tests for MountRefResolver (T4.2)."""

from __future__ import annotations

from pathlib import Path

from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.mount_ref_resolver import MountRefResolver
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import PathResolver
from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig


def _make_resolver(tmp_path: Path) -> tuple[MountRefResolver, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    ns = NamespaceConfig(root=str(root), default_mode="rw")
    fs_settings = FilesystemsSettings(namespaces={"artifacts": ns})

    class _FakeSettings:
        filesystems = fs_settings

    registry = NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]
    path_resolver = PathResolver(registry=registry)
    mount_resolver = MountRefResolver(registry=registry, resolver=path_resolver)
    return mount_resolver, root


class TestMountRefResolver:
    def test_empty_entries_returns_empty(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        result = resolver.resolve([])
        assert result == []

    def test_unknown_namespace_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="unknown",
            scope="org-1",
            path="file.mdx",
            target="/mnt/artifacts/file.mdx",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert result == []

    def test_file_entry_resolves_to_parent_directory(self, tmp_path: Path) -> None:
        """A file-level entry becomes a directory mount of the file's parent."""
        resolver, root = _make_resolver(tmp_path)
        (root / "org-1").mkdir()
        (root / "org-1" / "art-1").write_bytes(b"content")

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-1",
            target="/mnt/artifacts/org-1/art-1",
            mode="rw",
            etag="abc",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        mount = result[0]
        # Target is a directory (ends with "/"), source is the parent dir.
        assert mount.target == "/mnt/artifacts/org-1/"
        assert mount.source == root / "org-1"
        assert mount.access == "rw"
        assert mount.etag == "abc"

    def test_directory_entry_stays_directory(self, tmp_path: Path) -> None:
        """A directory-level entry keeps its directory target/source."""
        resolver, root = _make_resolver(tmp_path)
        (root / "org-1").mkdir()
        (root / "org-1" / "art-dir").mkdir()

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-dir",
            target="/mnt/artifacts/org-1/art-dir",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        assert result[0].target == "/mnt/artifacts/org-1/art-dir/"
        assert result[0].source == root / "org-1" / "art-dir"

    def test_ro_entry_produces_read_only_mount(self, tmp_path: Path) -> None:
        resolver, root = _make_resolver(tmp_path)
        (root / "org-1").mkdir()
        (root / "org-1" / "art-ro").write_bytes(b"data")

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-ro",
            target="/mnt/artifacts/org-1/art-ro",
            mode="ro",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        assert result[0].access == "ro"

    def test_nonexistent_path_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="does-not-exist",
            target="/mnt/artifacts/org-1/does-not-exist",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert result == []

    def test_invalid_traversal_path_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="artifacts",
            scope="../escape",
            path="file",
            target="/mnt/escape",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert result == []
