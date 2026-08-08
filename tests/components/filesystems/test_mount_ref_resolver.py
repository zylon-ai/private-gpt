"""Tests for MountRefResolver."""

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
        assert mount.host_path == root / "org-1"
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
        assert result[0].host_path == root / "org-1" / "art-dir"

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

    def test_uri_mount_uses_target_filename(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="artifacts",
            scope="artifact-1",
            path="_index.mdx",
            target="/mnt/artifacts/artifact-1.mdx",
            mode="ro",
            uri="file:///storage/_index.mdx",
        )

        result = resolver.resolve([entry])

        assert len(result) == 1
        assert result[0].target == "/mnt/artifacts/"
        assert result[0].uri_source is not None
        assert result[0].uri_source.filename == "artifact-1.mdx"

    def test_same_directory_uri_files_are_combined_generically(
        self, tmp_path: Path
    ) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entries = [
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="_content.md",
                target="/mnt/artifacts/artifact-1/_content.md",
                mode="rw",
                uri="file:///storage/_content.md",
            ),
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="_index.mdx",
                target="/mnt/artifacts/artifact-1/_index.mdx",
                mode="ro",
                uri="file:///storage/_index.mdx",
            ),
        ]

        result = resolver.resolve(entries)

        assert len(result) == 1
        mount = result[0]
        assert mount.target == "/mnt/artifacts/artifact-1/"
        assert mount.access == "rw"
        assert mount.uri_source is not None
        assert mount.uri_source.sources == [
            ("file:///storage/_content.md", "_content.md"),
            ("file:///storage/_index.mdx", "_index.mdx"),
        ]

    def test_overlapping_directories_are_not_rewritten(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entries = [
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="artifact-1/_content.md",
                target="/mnt/artifacts/artifact-1/_content.md",
                uri="file:///storage/artifact-1/_content.md",
            ),
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="artifact-2/_content.md",
                target="/mnt/artifacts/artifact-2/_content.md",
                uri="file:///storage/artifact-2/_content.md",
            ),
        ]

        result = resolver.resolve(entries)

        assert {mount.target for mount in result} == {
            "/mnt/artifacts/artifact-1/",
            "/mnt/artifacts/artifact-2/",
        }
