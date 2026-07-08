import json
from typing import Any

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillVersionEntity,
)
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.tools.remote_execution import build_rebuild_metadata, deserialize_rebuild_kwarg
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.di import get_global_injector
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
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_load_skill_tool,
                {
                    "skill_filter": self._skill_filter,
                    "skill_injection_mode": self._skill_injection_mode,
                    "name": name,
                    "type": type,
                },
            ),
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
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_unload_skill_tool,
                {
                    "skill_filter": self._skill_filter,
                    "skill_injection_mode": self._skill_injection_mode,
                    "name": name,
                    "type": type,
                },
            ),
        )

    def build_list_skills(
        self,
        name: str = SKILL_LIST_TOOL_NAME,
        type: str = SKILL_LIST_TOOL_NAME + "_v1",
    ) -> ToolSpec:
        async def list_skills(
            page: int = 0, page_size: int = 20
        ) -> list[ResultContentBlockType]:
            versions = await self._versions()
            total = len(versions)
            start = page * page_size
            page_versions = versions[start : start + page_size]
            return _ok(
                {
                    "skills": [
                        {
                            "name": s.frontmatter.name,
                            "description": s.frontmatter.description,
                            "skill_id": s.skill_id,
                            "version": s.version,
                        }
                        for s in page_versions
                    ],
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "has_more": start + page_size < total,
                }
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description="Browse the skill catalog (paginated). Use page/page_size to navigate large catalogs.",
            async_fn=list_skills,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_list_skills_tool,
                {
                    "skill_filter": self._skill_filter,
                    "skill_injection_mode": self._skill_injection_mode,
                    "name": name,
                    "type": type,
                },
            ),
        )


def _builder(
    skill_filter: SkillFilter,
    skill_injection_mode: str,
) -> "SkillManagementToolBuilder":
    injector = get_global_injector()
    return SkillManagementToolBuilder(
        skill_service=injector.get(SkillService),
        skill_filter=skill_filter,
        skill_injection_mode=skill_injection_mode,
    )


def rebuild_load_skill_tool(
    skill_filter: SkillFilter,
    skill_injection_mode: str,
    name: str = SKILL_LOAD_TOOL_NAME,
    type: str = SKILL_LOAD_TOOL_NAME + "_v1",
) -> ToolSpec:
    skill_filter = deserialize_rebuild_kwarg(skill_filter, SkillFilter)
    return _builder(skill_filter, skill_injection_mode).build_load_skill(
        name=name,
        type=type,
    )


def rebuild_unload_skill_tool(
    skill_filter: SkillFilter,
    skill_injection_mode: str,
    name: str = SKILL_UNLOAD_TOOL_NAME,
    type: str = SKILL_UNLOAD_TOOL_NAME + "_v1",
) -> ToolSpec:
    skill_filter = deserialize_rebuild_kwarg(skill_filter, SkillFilter)
    return _builder(skill_filter, skill_injection_mode).build_unload_skill(
        name=name,
        type=type,
    )


def rebuild_list_skills_tool(
    skill_filter: SkillFilter,
    skill_injection_mode: str,
    name: str = SKILL_LIST_TOOL_NAME,
    type: str = SKILL_LIST_TOOL_NAME + "_v1",
) -> ToolSpec:
    skill_filter = deserialize_rebuild_kwarg(skill_filter, SkillFilter)
    return _builder(skill_filter, skill_injection_mode).build_list_skills(
        name=name,
        type=type,
    )
