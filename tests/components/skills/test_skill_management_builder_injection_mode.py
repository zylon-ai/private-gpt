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
from private_gpt.events.models import TextBlock


def _version() -> SkillVersionEntity:
    return SkillVersionEntity(
        id="skillver_1",
        skill_id="skill_1",
        version="1000000",
        frontmatter=SkillFrontmatter(name="my-skill", description="Test skill"),
        storage_prefix="skills/tenant/skill_1/1000000",
        created_at=datetime.now(tz=UTC),
    )


def _resolved() -> SkillVersionWithSkillEntity:
    return SkillVersionWithSkillEntity(
        skill=SkillEntity(
            id="skill_1",
            collection="tenant-a",
            display_title="My Skill",
            source="custom",
            loading="lazy",
            readonly=False,
            latest_version="1000000",
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
        ),
        version=_version(),
    )


def _parse(result: list[TextBlock]) -> dict:
    block = result[0]
    return json.loads(block.text)


@pytest.mark.asyncio
async def test_load_skill_tool_result_mode_embeds_instructions() -> None:
    service = SimpleNamespace(
        recover_versions=AsyncMock(return_value=[_resolved()]),
        get_skill_body=AsyncMock(return_value="Detailed instructions"),
    )
    builder = SkillManagementToolBuilder(
        skill_service=service,
        skill_filter=SkillFilter(collection="tenant-a", skill_or_version_ids=None),
        skill_injection_mode="tool_result",
    )

    result = await builder.build_load_skill().async_fn(name="my-skill")
    payload = _parse(result)

    assert payload["loaded"] is True
    assert payload["instructions"] == "Detailed instructions"


@pytest.mark.asyncio
async def test_load_skill_system_prompt_mode_skips_instructions() -> None:
    service = SimpleNamespace(
        recover_versions=AsyncMock(return_value=[_resolved()]),
        get_skill_body=AsyncMock(return_value="Detailed instructions"),
    )
    builder = SkillManagementToolBuilder(
        skill_service=service,
        skill_filter=SkillFilter(collection="tenant-a", skill_or_version_ids=None),
        skill_injection_mode="system_prompt",
    )

    result = await builder.build_load_skill().async_fn(name="my-skill")
    payload = _parse(result)

    assert payload["loaded"] is True
    assert "instructions" not in payload
