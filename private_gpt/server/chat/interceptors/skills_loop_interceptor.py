import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.context.models.context_layer import (
    SkillBodyLayer,
    SkillCatalogEntry,
    SkillCatalogLayer,
)
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.server.utils.artifact_input import ArtifactType, SkillArtifact
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.skills.models.skill_entities import SkillVersionEntity

logger = logging.getLogger(__name__)

_SKILL_TOOL_CALL_NAMES = {SKILL_LOAD_TOOL_NAME, SKILL_UNLOAD_TOOL_NAME}


def _resolve_active_skill_names(
    conversation: list[ChatMessage],
    maximum_loaded_skills: int | None = None,
) -> set[str]:
    """Scan conversation for load_skill/unload_skill tool results.

    A skill is active if load_skill was called for it and unload_skill was NOT
    called after the most recent load.
    """
    loaded_order: list[str] = []
    for msg in conversation:
        if msg.role != MessageRole.TOOL:
            continue
        call_name = msg.additional_kwargs.get("tool_call_name")
        if call_name not in _SKILL_TOOL_CALL_NAMES:
            continue
        try:
            data = json.loads(msg.content or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        skill_name = data.get("name")
        if not skill_name:
            continue
        if call_name == SKILL_LOAD_TOOL_NAME and "error" not in data:
            if skill_name in loaded_order:
                loaded_order.remove(skill_name)
            loaded_order.append(skill_name)
            if (
                maximum_loaded_skills is not None
                and len(loaded_order) > maximum_loaded_skills
            ):
                loaded_order.pop(0)
        elif call_name == SKILL_UNLOAD_TOOL_NAME and data.get("unloaded"):
            if skill_name in loaded_order:
                loaded_order.remove(skill_name)
    return set(loaded_order)


@singleton
class SkillsInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(self, skill_service: SkillService, settings: Settings) -> None:
        self._skill_service = skill_service
        self._skill_injection_mode = settings.skills.skill_injection_mode

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        filter_input = self._find_skill_filter(state.input.request.tool_context)

        stack = state.input.context_stack
        stack = stack.remove_layers_of_type(LayerType.SKILL_CATALOG)
        stack = stack.remove_layers_of_type(LayerType.SKILL_BODY)
        if filter_input is None:
            state.input.context_stack = stack
            context.set_state(state)
            return

        skills_cache = state.runtime.cache.skill
        entries = skills_cache.entries if skills_cache else []
        if not entries:
            state.input.context_stack = stack
            context.set_state(state)
            return

        active_from_tools = _resolve_active_skill_names(
            state.input.request.messages,
            maximum_loaded_skills=state.input.request.context.maximum_loaded_skills,
        )
        active_versions: list[SkillVersionEntity] = []
        catalog_entries: list[SkillCatalogEntry] = []

        for entry in entries:
            skill = entry.skill
            version = entry.version
            name = version.frontmatter.name
            loading = skill.loading
            if loading == "eager":
                active_versions.append(version)
            elif name in active_from_tools:
                if self._skill_injection_mode == "system_prompt":
                    active_versions.append(version)
            else:
                catalog_entries.append(
                    SkillCatalogEntry(
                        id=version.skill_id,
                        name=name,
                        description=version.frontmatter.description,
                        loading=loading,
                    )
                )

        if catalog_entries:
            stack = stack.append_layer(
                SkillCatalogLayer(entries=catalog_entries, source="skills")
            )

        for version in active_versions:
            try:
                instructions = await self._skill_service.get_skill_body(version)
            except Exception as exc:
                logger.warning(
                    "Skills: unable to load body for %s: %s", version.skill_id, exc
                )
                continue
            stack = stack.append_layer(
                SkillBodyLayer(
                    skill_id=version.skill_id,
                    name=version.frontmatter.name,
                    version=version.version,
                    instructions=instructions,
                    source=f"skill:{version.frontmatter.name}",
                )
            )

        state.input.context_stack = stack
        context.set_state(state)

    def _find_skill_filter(
        self, tool_context: Sequence[ArtifactType]
    ) -> SkillFilter | None:
        for artifact in tool_context:
            if isinstance(artifact, SkillArtifact):
                return artifact.skill_filter
        return None
