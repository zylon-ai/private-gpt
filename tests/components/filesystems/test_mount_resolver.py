"""Tests for MountResolver — one entry, one exact bind mount."""

from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.mount_resolver import MountResolver
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import PathResolver
from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig

if TYPE_CHECKING:
    from pathlib import Path


def _make_resolver(
    tmp_path: Path, *, hydration: bool = False
) -> tuple[MountResolver, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    ns = NamespaceConfig(root=str(root), default_mode="rw", hydration=hydration)
    fs_settings = FilesystemsSettings(namespaces={"artifacts": ns})

    class _FakeSettings:
        filesystems = fs_settings

    registry = NamespaceRegistry(settings=_FakeSettings())  # type: ignore[arg-type]
    path_resolver = PathResolver(registry=registry)
    mount_resolver = MountResolver(registry=registry, resolver=path_resolver)
    return mount_resolver, root


class TestMountResolver:
    def test_empty_entries_returns_empty(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        assert resolver.resolve([]) == []

    def test_unknown_namespace_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="unknown",
            scope="org-1",
            path="file.mdx",
            target="/mnt/artifacts/file.mdx",
            mode="rw",
        )
        assert resolver.resolve([entry]) == []

    def test_file_entry_keeps_exact_target_and_file_source(
        self, tmp_path: Path
    ) -> None:
        """A file-level entry becomes a file mount: exact target, exact host file."""
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
        assert mount.target == "/mnt/artifacts/org-1/art-1"
        assert mount.host_path == root / "org-1" / "art-1"
        assert mount.access == "rw"
        assert mount.etag == "abc"
        assert mount.is_folder is False

    def test_entry_target_is_never_rewritten(self, tmp_path: Path) -> None:
        """The target decides the mount kind; the resolver never adds or moves
        trailing slashes — the Backend owns the exact target."""
        resolver, root = _make_resolver(tmp_path)
        (root / "org-1").mkdir()
        (root / "org-1" / "art-dir").mkdir()

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-dir",
            target="/mnt/artifacts/org-1/art-dir/",
            mode="rw",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        mount = result[0]
        assert mount.target == "/mnt/artifacts/org-1/art-dir/"
        assert mount.host_path == root / "org-1" / "art-dir"
        assert mount.is_folder is True

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

    def test_missing_path_without_uri_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="does-not-exist",
            target="/mnt/artifacts/org-1/does-not-exist",
            mode="rw",
        )
        assert resolver.resolve([entry]) == []

    def test_missing_path_with_uri_is_kept_only_when_hydration_enabled(
        self, tmp_path: Path
    ) -> None:
        """URI-backed missing hosts are kept only when hydration can fill them."""
        resolver, root = _make_resolver(tmp_path, hydration=True)
        (root / "org-1").mkdir()

        entry = MountEntry(
            namespace="artifacts",
            scope="org-1",
            path="art-1/_content.md",
            target="/home/agent/workspace/potato.md",
            mode="rw",
            etag="abc",
            uri="file:///storage/_content.md",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        mount = result[0]
        assert mount.target == "/home/agent/workspace/potato.md"
        assert mount.host_path == root / "org-1" / "art-1" / "_content.md"
        assert mount.uri_source is not None
        assert mount.etag == "abc"

    def test_missing_path_with_uri_is_skipped_when_hydration_disabled(
        self, tmp_path: Path
    ) -> None:
        """Missing hosts must not be bind-mounted (would create empty dirs)."""
        resolver, _ = _make_resolver(tmp_path, hydration=False)
        entry = MountEntry(
            namespace="artifacts",
            scope="thread-1",
            path="art-1/_content.md",
            target="/mnt/artifacts/art-1/_content.md",
            mode="rw",
            uri="s3://private/missing/art-1/_content.md",
        )
        assert resolver.resolve([entry]) == []

    def test_s3_uri_resolves_host_path_from_object_key(self, tmp_path: Path) -> None:
        """URI object keys map under the namespace root, not thread/scope."""
        resolver, root = _make_resolver(tmp_path)
        host = root / "00000000-org" / "proj" / "art-1" / "_content.md"
        host.parent.mkdir(parents=True)
        host.write_bytes(b"content")

        entry = MountEntry(
            namespace="artifacts",
            scope="thread-does-not-exist-on-disk",
            path="art-1/_content.md",
            target="/mnt/artifacts/art-1/_content.md",
            mode="rw",
            uri="s3://private/00000000-org/proj/art-1/_content.md",
        )
        result = resolver.resolve([entry])
        assert len(result) == 1
        mount = result[0]
        assert mount.host_path == host
        assert mount.target == "/mnt/artifacts/art-1/_content.md"
        assert mount.is_folder is False
        assert mount.host_path.is_file()

    def test_invalid_traversal_path_is_skipped(self, tmp_path: Path) -> None:
        resolver, _ = _make_resolver(tmp_path)
        entry = MountEntry(
            namespace="artifacts",
            scope="../escape",
            path="file",
            target="/mnt/escape",
            mode="rw",
        )
        assert resolver.resolve([entry]) == []

    def test_uri_file_mount_keeps_exact_target(self, tmp_path: Path) -> None:
        """No more parent-directory mangling: a file entry stays a file mount."""
        resolver, _ = _make_resolver(tmp_path, hydration=True)
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
        assert result[0].target == "/mnt/artifacts/artifact-1.mdx"
        assert result[0].uri_source is not None
        assert result[0].uri_source.filename == "artifact-1.mdx"

    def test_volume_names_are_unique_within_a_scope(self, tmp_path: Path) -> None:
        """Two artifacts of the same thread must not produce duplicate volume
        names (sandbox backends reject duplicate volume names)."""
        resolver, _ = _make_resolver(tmp_path, hydration=True)
        entries = [
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="art-1/_content.md",
                target="/home/agent/workspace/a.md",
                mode="rw",
                uri="file:///storage/_content.md",
            ),
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="art-2/_content.md",
                target="/home/agent/workspace/b.md",
                mode="rw",
                uri="file:///storage/_content.md",
            ),
        ]

        mounts = resolver.resolve(entries)

        assert len(mounts) == 2
        assert mounts[0].name != mounts[1].name
        assert all(m.name.startswith("mount-artifacts-") for m in mounts)

    def test_two_uri_files_are_not_merged(self, tmp_path: Path) -> None:
        """Each entry is its own mount; merging is the runtime's job (none)."""
        resolver, _ = _make_resolver(tmp_path, hydration=True)
        entries = [
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="artifact-1/_content.md",
                target="/mnt/artifacts/artifact-1/_content.md",
                mode="rw",
                uri="file:///storage/_content.md",
            ),
            MountEntry(
                namespace="artifacts",
                scope="thread-1",
                path="artifact-1/_index.mdx",
                target="/mnt/artifacts/artifact-1/_index.mdx",
                mode="ro",
                uri="file:///storage/_index.mdx",
            ),
        ]

        result = resolver.resolve(entries)

        assert {mount.target for mount in result} == {
            "/mnt/artifacts/artifact-1/_content.md",
            "/mnt/artifacts/artifact-1/_index.mdx",
        }
