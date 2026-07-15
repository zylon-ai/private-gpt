from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.storage.models import StoredFile
from private_gpt.settings.settings import unsafe_typed_settings


def _skill_md_bytes() -> bytes:
    return (
        b"---\nname: test-skill\ndescription: test description\n---\n\nUse this skill\n"
    )


def _settings():
    settings = unsafe_typed_settings.model_copy(deep=True)
    settings.skills.storage_provider = "local"
    settings.s3.durable_bucket_name = "test-bucket"
    return settings


def _service(*, repo: object, storage: object) -> SkillService:
    return SkillService(
        settings=_settings(),
        storage_component=SimpleNamespace(get_object_storage=lambda **_: storage),
        skill_repository=repo,
    )


@pytest.mark.asyncio
async def test_create_skill_rolls_back_storage_when_repository_fails() -> None:
    storage = SimpleNamespace(
        write_bundle=AsyncMock(),
        delete_prefix=AsyncMock(),
    )
    repo = SimpleNamespace(
        create_skill=AsyncMock(side_effect=RuntimeError("db write failed"))
    )
    service = _service(repo=repo, storage=storage)

    with pytest.raises(RuntimeError, match="db write failed"):
        await service.create_skill(
            collection="tenant-a",
            display_title="Skill",
            source="custom",
            loading="lazy",
            readonly=False,
            files=[
                StoredFile(
                    path="SKILL.md",
                    content=_skill_md_bytes(),
                    mime_type="text/markdown",
                )
            ],
        )

    assert storage.delete_prefix.await_count == 1


@pytest.mark.asyncio
async def test_create_version_rolls_back_storage_when_repository_fails() -> None:
    storage = SimpleNamespace(
        write_bundle=AsyncMock(),
        delete_prefix=AsyncMock(),
    )
    repo = SimpleNamespace(
        create_version=AsyncMock(side_effect=RuntimeError("db write failed"))
    )
    service = _service(repo=repo, storage=storage)

    with pytest.raises(RuntimeError, match="db write failed"):
        await service.create_version(
            skill_id="skill_1",
            collection="tenant-a",
            files=[
                StoredFile(
                    path="SKILL.md",
                    content=_skill_md_bytes(),
                    mime_type="text/markdown",
                )
            ],
        )

    assert storage.delete_prefix.await_count == 1


@pytest.mark.asyncio
async def test_storage_prefix_uses_version_id_not_version_token() -> None:
    storage = SimpleNamespace(
        write_bundle=AsyncMock(),
        delete_prefix=AsyncMock(),
    )
    repo = SimpleNamespace(create_skill=AsyncMock())
    service = _service(repo=repo, storage=storage)

    await service.create_skill(
        collection="tenant-a",
        display_title="Skill",
        source="custom",
        loading="lazy",
        readonly=False,
        files=[
            StoredFile(
                path="SKILL.md",
                content=_skill_md_bytes(),
                mime_type="text/markdown",
            )
        ],
    )

    payload = repo.create_skill.await_args.args[0]
    assert payload.initial_version.id in payload.initial_version.storage_prefix
