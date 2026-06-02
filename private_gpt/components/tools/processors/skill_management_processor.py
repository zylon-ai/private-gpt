from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.tools.builders.skill_management_builder import (
    SkillManagementToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _get_tool_context,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
    _wrapper_tool,
)
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
    SKILLS_TOOL_NAME,
)
from private_gpt.server.utils.artifact_input import SkillArtifact
from private_gpt.settings.settings import Settings


@singleton
class SkillManagementProcessor(ToolProcessor):
    @inject
    def __init__(self, settings: Settings, skill_service: SkillService) -> None:
        self._settings = settings
        self._skill_service = skill_service

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(
                tool,
                SKILLS_TOOL_NAME,
                SKILL_LOAD_TOOL_NAME,
                SKILL_UNLOAD_TOOL_NAME,
                SKILL_LIST_TOOL_NAME,
            ) or not _is_unresolved_tool(tool):
                continue

            tool_context = _get_tool_context(request, tool)
            skill_artifacts = [a for a in tool_context if isinstance(a, SkillArtifact)]
            if not skill_artifacts:
                raise ValueError(
                    "Skill management tools require a SkillArtifact in the tool context."
                )
            if len(skill_artifacts) > 1:
                raise ValueError(
                    "Only one SkillArtifact is supported per skill management tool."
                )

            builder = SkillManagementToolBuilder(
                skill_service=self._skill_service,
                skill_filter=skill_artifacts[0].skill_filter,
                skill_injection_mode=self._settings.skills.skill_injection_mode,
            )

            if _tool_matches(tool, SKILLS_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        _wrapper_tool(
                            SKILL_LOAD_TOOL_NAME,
                        ),
                        _wrapper_tool(
                            SKILL_UNLOAD_TOOL_NAME,
                        ),
                        _wrapper_tool(
                            SKILL_LIST_TOOL_NAME,
                        ),
                    ],
                )
            if _tool_matches(tool, SKILL_LOAD_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        builder.build_load_skill(
                            name=tool.name or SKILL_LOAD_TOOL_NAME,
                            type=tool.type or SKILL_LOAD_TOOL_NAME + "_v1",
                        )
                    ],
                )
            if _tool_matches(tool, SKILL_UNLOAD_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        builder.build_unload_skill(
                            name=tool.name or SKILL_UNLOAD_TOOL_NAME,
                            type=tool.type or SKILL_UNLOAD_TOOL_NAME + "_v1",
                        )
                    ],
                )
            if _tool_matches(tool, SKILL_LIST_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        builder.build_list_skills(
                            name=tool.name or SKILL_LIST_TOOL_NAME,
                            type=tool.type or SKILL_LIST_TOOL_NAME + "_v1",
                        )
                    ],
                )
        return False
