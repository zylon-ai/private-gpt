"""SkillLoader host-path resolution for namespace-backed mounts."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from private_gpt.components.skills.models.skill_entities import (
    SkillFrontmatter,
    SkillVersionEntity,
)
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_loader import SkillLoader

if TYPE_CHECKING:
    from pathlib import Path


class _FakeRegistry:
    def __init__(self, root: Path) -> None:
        self._root = root

    def root(self, namespace: str) -> Path:
        if namespace != "skills":
            raise KeyError(namespace)
        return self._root


def _loader(tmp_path: Path, registry=None) -> SkillLoader:
    settings = SimpleNamespace(
        data=SimpleNamespace(local_data_folder=str(tmp_path / "data")),
        skills=SimpleNamespace(storage_provider="local"),
        s3=SimpleNamespace(durable_bucket_name="storage"),
    )
    storage_component = MagicMock()
    storage_component.get_object_storage.return_value = MagicMock()
    return SkillLoader(
        settings=settings,
        storage_component=storage_component,
        skill_service=MagicMock(),
        namespace_registry=registry or _FakeRegistry(tmp_path / "skills-root"),
    )


def _version() -> SkillVersionEntity:
    return SkillVersionEntity(
        id="skillver_abc",
        skill_id="skill_xlsx",
        version="1",
        storage_prefix="skills/00000000-0000-7000-8001-000000000001/skill_xlsx/skillver_abc",
        frontmatter=SkillFrontmatter(name="xlsx", description="xlsx skill"),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_mounts_for_versions_uses_storage_prefix_under_skills_root(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path)
    version = _version()

    mounts = loader.mounts_for_versions([version])

    assert len(mounts) == 1
    mount = mounts[0]
    assert mount.target == skill_mount_path("xlsx")
    assert mount.host_path == (
        tmp_path
        / "skills-root"
        / "skills"
        / "00000000-0000-7000-8001-000000000001"
        / "skill_xlsx"
        / "skillver_abc"
    )
    # Must not use the bare version id (storage layout is storage_prefix).
    assert mount.host_path != tmp_path / "skills-root" / version.id
    assert mount.source is not None
    assert mount.source.namespace == "skills"
    assert mount.uri_source is not None
    assert mount.uri_source.uri == version.storage_prefix


def test_mounts_for_versions_without_skills_namespace_has_no_host_path(
    tmp_path: Path,
) -> None:
    class EmptyRegistry:
        def root(self, namespace: str) -> Path:
            raise KeyError(namespace)

    loader = _loader(tmp_path, registry=EmptyRegistry())
    mounts = loader.mounts_for_versions([_version()])

    assert mounts[0].host_path is None
    assert mounts[0].target == "/mnt/skills/xlsx/"
