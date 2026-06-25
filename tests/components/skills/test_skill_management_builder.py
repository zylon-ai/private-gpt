import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.skills.models.skill_entities import (
    SkillEntity,
    SkillFilter,
    SkillFrontmatter,
    SkillVersionEntity,
    SkillVersionWithSkillEntity,
)
from private_gpt.components.tools.builders.skill_management_builder import (
    SkillManagementToolBuilder,
)
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import TextBlock


def _parse(result: list[TextBlock]) -> dict:
    assert result
    block = result[0]
    assert isinstance(block, TextBlock)
    return json.loads(block.text)


def _resolved(
    *,
    version_id: str = "skillver_1",
    skill_id: str = "skill_1",
    version: str = "1000000",
    name: str = "my-skill",
    description: str = "Test skill",
) -> SkillVersionWithSkillEntity:
    version_entity = SkillVersionEntity(
        id=version_id,
        skill_id=skill_id,
        version=version,
        frontmatter=SkillFrontmatter(name=name, description=description),
        storage_prefix="skills/tenant/skill_1/skillver_1",
        created_at=datetime.now(tz=UTC),
    )
    skill_entity = SkillEntity(
        id=skill_id,
        collection="tenant",
        display_title="My Skill",
        source="custom",
        loading="lazy",
        readonly=False,
        latest_version=version,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
    )
    return SkillVersionWithSkillEntity(skill=skill_entity, version=version_entity)


@pytest.fixture
def builder() -> SkillManagementToolBuilder:
    service = SimpleNamespace(
        recover_versions=AsyncMock(return_value=[_resolved()]),
        get_skill_body=AsyncMock(return_value="Skill body"),
    )
    return SkillManagementToolBuilder(
        skill_service=service,
        skill_filter=SkillFilter(collection="tenant", skill_or_version_ids=None),
    )


def test_build_tool_types(builder: SkillManagementToolBuilder) -> None:
    assert builder.build_load_skill().type == "load_skill_v1"
    assert builder.build_unload_skill().type == "unload_skill_v1"
    assert builder.build_list_skills().type == "list_skills_v1"


def test_build_custom_name_and_type(builder: SkillManagementToolBuilder) -> None:
    spec = builder.build_load_skill(name="my_load", type="my_load_v2")
    assert spec.name == "my_load"
    assert spec.type == "my_load_v2"


@pytest.mark.asyncio
async def test_load_skill_returns_payload(builder: SkillManagementToolBuilder) -> None:
    result = await builder.build_load_skill().async_fn(name="my-skill")
    data = _parse(result)
    assert data == {
        "name": "my-skill",
        "skill_id": "skill_1",
        "version": "1000000",
        "loaded": True,
    }


@pytest.mark.asyncio
async def test_load_skill_missing_returns_error(
    builder: SkillManagementToolBuilder,
) -> None:
    result = await builder.build_load_skill().async_fn(name="unknown")
    data = _parse(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_unload_skill_returns_payload(
    builder: SkillManagementToolBuilder,
) -> None:
    result = await builder.build_unload_skill().async_fn(name="my-skill")
    data = _parse(result)
    assert data == {"name": "my-skill", "unloaded": True}


@pytest.mark.asyncio
async def test_list_skills_returns_resolved_versions(
    builder: SkillManagementToolBuilder,
) -> None:
    result = await builder.build_list_skills().async_fn()
    data = _parse(result)
    assert data == {
        "skills": [
            {
                "name": "my-skill",
                "description": "Test skill",
                "skill_id": "skill_1",
                "version": "1000000",
            }
        ],
        "page": 0,
        "page_size": 20,
        "total": 1,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_list_skills_empty_when_no_versions() -> None:
    service = SimpleNamespace(
        recover_versions=AsyncMock(return_value=[]),
        get_skill_body=AsyncMock(return_value="Skill body"),
    )
    builder = SkillManagementToolBuilder(
        skill_service=service,
        skill_filter=SkillFilter(collection="tenant", skill_or_version_ids=None),
    )
    result = await builder.build_list_skills().async_fn()
    data = _parse(result)
    assert data == {
        "skills": [],
        "page": 0,
        "page_size": 20,
        "total": 0,
        "has_more": False,
    }


def test_tool_names_match_constraints(builder: SkillManagementToolBuilder) -> None:
    assert builder.build_load_skill().name == SKILL_LOAD_TOOL_NAME
    assert builder.build_unload_skill().name == SKILL_UNLOAD_TOOL_NAME
    assert builder.build_list_skills().name == SKILL_LIST_TOOL_NAME
