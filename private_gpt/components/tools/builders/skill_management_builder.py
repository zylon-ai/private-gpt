import json
from typing import Any

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillVersionEntity,
)
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import ResultContentBlockType, TextBlock


def _ok(data: Any) -> list[ResultContentBlockType]:
    return [TextBlock(text=json.dumps(data, default=str))]


def _error(msg: str) -> list[ResultContentBlockType]:
    return [TextBlock(text=json.dumps({"error": msg}))]


class SkillManagementToolBuilder:
    """Build skill-management ToolSpec instances from SkillService + SkillFilter."""

    def __init__(
        self,
        skill_service: SkillService,
        skill_filter: SkillFilter,
        skill_injection_mode: str = "system_prompt",
    ) -> None:
        self._skill_service = skill_service
        self._skill_filter = skill_filter
        self._skill_injection_mode = skill_injection_mode

    async def _versions(self) -> list[SkillVersionEntity]:
        resolved = await self._skill_service.recover_versions(self._skill_filter)
        return [item.version for item in resolved]

    def build_load_skill(
        self,
        name: str = SKILL_LOAD_TOOL_NAME,
        type: str = SKILL_LOAD_TOOL_NAME + "_v1",
    ) -> ToolSpec:
        async def load_skill(name: str) -> list[ResultContentBlockType]:
            versions = await self._versions()
            for version in versions:
                if version.frontmatter.name == name:
                    payload: dict[str, Any] = {
                        "name": version.frontmatter.name,
                        "skill_id": version.skill_id,
                        "version": version.version,
                        "loaded": True,
                    }
                    if self._skill_injection_mode == "tool_result":
                        payload[
                            "instructions"
                        ] = await self._skill_service.get_skill_body(version)
                    return _ok(payload)
            return _error(f"Skill '{name}' not found in current skill_filter")

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description="Mark one available skill as loaded for this conversation.",
            async_fn=load_skill,
        )

    def build_unload_skill(
        self,
        name: str = SKILL_UNLOAD_TOOL_NAME,
        type: str = SKILL_UNLOAD_TOOL_NAME + "_v1",
    ) -> ToolSpec:
        async def unload_skill(name: str) -> list[ResultContentBlockType]:
            return _ok({"name": name, "unloaded": True})

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description="Mark one loaded skill as unloaded for this conversation.",
            async_fn=unload_skill,
        )

    def build_list_skills(
        self,
        name: str = SKILL_LIST_TOOL_NAME,
        type: str = SKILL_LIST_TOOL_NAME + "_v1",
    ) -> ToolSpec:
        async def list_skills() -> list[ResultContentBlockType]:
            versions = await self._versions()
            return _ok(
                {
                    "skills": [
                        {
                            "name": s.frontmatter.name,
                            "description": s.frontmatter.description,
                            "skill_id": s.skill_id,
                            "version": s.version,
                        }
                        for s in versions
                    ]
                }
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description="List available skills from current skill_filter.",
            async_fn=list_skills,
        )
