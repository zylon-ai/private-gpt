"""End-to-end mount pipeline: MountEntry shape -> host bind volumes.

Host paths are always exact:

- artifacts: ``{artifacts_root}/{s3_object_key}`` when a URI is present
- skills: ``{skills_root}/{storage_prefix}``
- session layout: only user/uploads/outputs per session id

When hydration is enabled, missing host files are materialized from the URI
into those same paths before the sandbox is created. When hydration is off,
missing hosts are skipped so bind-mounts never create empty directories.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from private_gpt.components.environment.hydration import HydratingEnvironmentManager
from private_gpt.components.environment.layout import DEFAULT_SESSION_LAYOUT
from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.environment.mounter import LocalDirMounter
from private_gpt.components.filesystems.mount_entry import MountEntry
from private_gpt.components.filesystems.mount_resolver import MountResolver
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.filesystems.path_resolver import PathResolver
from private_gpt.components.sandbox.mount import MountFile, UriSource
from private_gpt.components.skills.models.skill_entities import (
    SkillFrontmatter,
    SkillVersionEntity,
)
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_loader import SkillLoader
from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig

if TYPE_CHECKING:
    from private_gpt.components.sandbox.mount import Mount


class RecordingProvider:
    def __init__(self) -> None:
        self.volumes: list[list[Mount] | None] = []
        self._counter = 0

    async def restore_session(self, *args, **kwargs):
        return None

    async def create_session(
        self,
        user_id=None,
        timeout=None,
        bundle_specs=None,
        *,
        session_id=None,
        volumes=None,
        env=None,
        fingerprint=None,
    ):
        self._counter += 1
        self.volumes.append(volumes)
        sandbox = SimpleNamespace(sandbox_id=f"sb-{self._counter}", killed=False)

        async def make_dir(canonical: str, **kwargs) -> None:
            return None

        sandbox.make_dir = make_dir
        return sandbox

    async def renew_session(self, session) -> None:
        return None

    async def kill_session(self, session, session_id=None) -> None:
        session.killed = True


def _ns_settings(tmp_path: Path, *, hydration: bool) -> SimpleNamespace:
    root_artifacts = tmp_path / "artifacts"
    root_skills = tmp_path / "skills"
    root_session = tmp_path / "sessions"
    for root in (root_artifacts, root_skills, root_session):
        root.mkdir(parents=True, exist_ok=True)
    fs = FilesystemsSettings(
        namespaces={
            "session": NamespaceConfig(root=str(root_session), default_mode="rw"),
            "artifacts": NamespaceConfig(
                root=str(root_artifacts), default_mode="rw", hydration=hydration
            ),
            "skills": NamespaceConfig(
                root=str(root_skills), default_mode="ro", hydration=hydration
            ),
        }
    )
    return SimpleNamespace(filesystems=fs)


def _resolver(
    tmp_path: Path, *, hydration: bool
) -> tuple[MountResolver, Path, Path, Path]:
    settings = _ns_settings(tmp_path, hydration=hydration)
    registry = NamespaceRegistry(settings=settings)  # type: ignore[arg-type]
    resolver = MountResolver(
        registry=registry, resolver=PathResolver(registry=registry)
    )
    return (
        resolver,
        Path(settings.filesystems.namespaces["artifacts"].root),
        Path(settings.filesystems.namespaces["skills"].root),
        Path(settings.filesystems.namespaces["session"].root),
    )


@pytest.mark.asyncio
async def test_artifact_skill_and_session_exact_mounts(tmp_path: Path) -> None:
    """Content already on disk under the real storage keys is bind-mounted exactly."""
    resolver, artifacts_root, skills_root, session_root = _resolver(
        tmp_path, hydration=False
    )

    content = artifacts_root / "00000000-org" / "proj" / "019f0488-art" / "_content.md"
    projection = content.with_name("_index.mdx")
    content.parent.mkdir(parents=True)
    content.write_bytes(b"# real content")
    projection.write_bytes(b"export const meta = {}")

    skill_host = (
        skills_root
        / "skills"
        / "00000000-0000-7000-8001-000000000001"
        / "skill_xlsx"
        / "skillver_1"
    )
    skill_host.mkdir(parents=True)
    (skill_host / "SKILL.md").write_text("---\nname: xlsx\ndescription: x\n---\n")

    artifact_entries = [
        MountEntry(
            namespace="artifacts",
            scope="thread-abc",
            path="019f0488-art/_content.md",
            target="/mnt/artifacts/019f0488-art/_content.md",
            mode="rw",
            uri="s3://private/00000000-org/proj/019f0488-art/_content.md",
        ),
        MountEntry(
            namespace="artifacts",
            scope="thread-abc",
            path="019f0488-art/_index.mdx",
            target="/mnt/artifacts/019f0488-art.mdx",
            mode="ro",
            uri="s3://private/00000000-org/proj/019f0488-art/_index.mdx",
        ),
    ]
    artifact_mounts = resolver.resolve(artifact_entries)
    assert len(artifact_mounts) == 2
    by_target = {m.target: m for m in artifact_mounts}
    assert by_target["/mnt/artifacts/019f0488-art/_content.md"].host_path == content
    assert by_target["/mnt/artifacts/019f0488-art.mdx"].host_path == projection
    assert content.is_file()
    assert not content.is_dir()
    assert projection.is_file()
    assert not projection.is_dir()

    ns_settings = _ns_settings(tmp_path, hydration=False)
    registry = NamespaceRegistry(settings=ns_settings)  # type: ignore[arg-type]
    loader = SkillLoader(
        settings=SimpleNamespace(
            data=SimpleNamespace(local_data_folder=str(tmp_path / "data")),
            skills=SimpleNamespace(storage_provider="local"),
            s3=SimpleNamespace(durable_bucket_name="storage"),
        ),
        storage_component=MagicMock(
            get_object_storage=MagicMock(return_value=MagicMock())
        ),
        skill_service=MagicMock(),
        namespace_registry=registry,
    )
    version = SkillVersionEntity(
        id="skillver_1",
        skill_id="skill_xlsx",
        version="1",
        storage_prefix=(
            "skills/00000000-0000-7000-8001-000000000001/skill_xlsx/skillver_1"
        ),
        frontmatter=SkillFrontmatter(name="xlsx", description="xlsx skill"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    skill_mounts = loader.mounts_for_versions([version])
    assert skill_mounts[0].target == skill_mount_path("xlsx")
    assert skill_mounts[0].host_path == skill_host

    # Session layout: only the 3 exact folders, never the namespace root.
    layout = LocalDirMounter(session_root)
    session_volumes = layout.session_volumes("session-1")
    assert [(v.name, v.target) for v in session_volumes] == [
        (m.name, m.target) for m in DEFAULT_SESSION_LAYOUT
    ]
    assert all(
        v.host_path == session_root / v.name / "session-1" for v in session_volumes
    )

    provider = RecordingProvider()
    manager = EnvironmentManager(
        sandbox_provider=provider,  # type: ignore[arg-type]
        layout_mounter=layout,
        ttl_seconds=3600,
        namespaces=ns_settings.filesystems.namespaces,
    )
    await manager.acquire(
        "session-1",
        mounts=artifact_mounts + skill_mounts,
    )

    volumes = provider.volumes[0] or []
    targets = {v.target for v in volumes}
    host_paths = {str(v.host_path) for v in volumes if v.host_path is not None}

    assert "/home/agent/workspace/" in targets
    assert "/mnt/user-data/uploads/" in targets
    assert "/mnt/user-data/outputs/" in targets
    assert "/mnt/artifacts/019f0488-art/_content.md" in targets
    assert "/mnt/artifacts/019f0488-art.mdx" in targets
    assert "/mnt/skills/xlsx/" in targets

    # Namespace roots themselves never mounted.
    assert "/mnt/artifacts/" not in targets
    assert "/mnt/skills/" not in targets
    assert "/mnt/session/" not in targets
    assert str(artifacts_root) not in host_paths
    assert str(skills_root) not in host_paths
    assert str(session_root) not in host_paths


@pytest.mark.asyncio
async def test_hydration_materializes_uri_into_object_key_host_path(
    tmp_path: Path,
) -> None:
    """With hydration on, missing hosts are filled at the URI object-key path."""
    resolver, artifacts_root, _skills_root, session_root = _resolver(
        tmp_path, hydration=True
    )

    entry = MountEntry(
        namespace="artifacts",
        scope="thread-abc",
        path="019f0488-art/_content.md",
        target="/home/agent/workspace/report.md",
        mode="rw",
        etag="etag-1",
        uri="s3://private/00000000-org/proj/019f0488-art/_content.md",
    )
    mounts = resolver.resolve([entry])
    assert len(mounts) == 1
    mount = mounts[0]
    expected_host = (
        artifacts_root / "00000000-org" / "proj" / "019f0488-art" / "_content.md"
    )
    assert mount.host_path == expected_host
    assert not expected_host.exists()

    async def fetch() -> list[MountFile]:
        return [MountFile(path="_content.md", content=b"# hydrated content")]

    mount.uri_source = UriSource(uri=entry.uri or "", fetch=fetch)

    provider = RecordingProvider()
    layout = LocalDirMounter(session_root)
    inner = EnvironmentManager(
        sandbox_provider=provider,  # type: ignore[arg-type]
        layout_mounter=layout,
        ttl_seconds=3600,
    )
    manager = HydratingEnvironmentManager(
        manager=inner,
        namespaces={
            "artifacts": NamespaceConfig(
                root=str(artifacts_root), default_mode="rw", hydration=True
            )
        },
    )

    await manager.acquire("session-1", mounts=[mount])

    assert expected_host.is_file()
    assert expected_host.read_bytes() == b"# hydrated content"
    assert not expected_host.is_dir()

    volumes = provider.volumes[0] or []
    artifact_vol = next(
        v for v in volumes if v.target == "/home/agent/workspace/report.md"
    )
    assert artifact_vol.host_path == expected_host
    assert {v.target for v in volumes} >= {
        "/home/agent/workspace/",
        "/mnt/user-data/uploads/",
        "/mnt/user-data/outputs/",
        "/home/agent/workspace/report.md",
    }


@pytest.mark.asyncio
async def test_missing_artifact_without_hydration_is_not_bind_mounted(
    tmp_path: Path,
) -> None:
    resolver, artifacts_root, _, _session_root = _resolver(tmp_path, hydration=False)
    entry = MountEntry(
        namespace="artifacts",
        scope="thread-abc",
        path="missing/_content.md",
        target="/mnt/artifacts/missing/_content.md",
        mode="rw",
        uri="s3://private/org/proj/missing/_content.md",
    )
    assert resolver.resolve([entry]) == []
    polluted = artifacts_root / "thread-abc" / "missing" / "_content.md"
    assert not polluted.exists()
