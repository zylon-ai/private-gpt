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

    def test_valid_existing_file_resolves_to_volume_spec(self, tmp_path: Path) -> None:
        resolver, root = _make_resolver(tmp_path)
        # Create the file
        (root / "org-1").mkdir()
        (root / "org-1" / "art-1").write_bytes(b"content")

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-1",
            target="/mnt/artifacts/org-1/art-1",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        assert str(result[0].mount_path) == "/mnt/artifacts/org-1/art-1"
        assert result[0].read_only is False

    def test_ro_entry_produces_read_only_volume(self, tmp_path: Path) -> None:
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
        assert result[0].read_only is True

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
