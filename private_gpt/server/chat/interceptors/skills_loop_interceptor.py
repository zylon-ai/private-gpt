import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.context.models.context_layer import (
    MountsLayer,
    SkillBodyLayer,
    SkillCatalogEntry,
    SkillCatalogLayer,
)
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_loader import SkillLoader
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


def _parse_tool_result_content(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_call_args(msg: ChatMessage) -> dict[str, Any]:
    args = msg.additional_kwargs.get("tool_call_args")
    return args if isinstance(args, dict) else {}


def _resolve_skill_states(
    conversation: list[ChatMessage],
    maximum_loaded_skills: int | None = None,
) -> tuple[set[str], set[str]]:
    """Scan tool call history and return (active, to_remove) skill name sets.

    active: currently loaded skills (load_skill called, no subsequent unload).
    to_remove: skills explicitly unloaded or evicted due to capacity limit.
    Skills never interacted with appear in neither set.

    Prefer the tool-result JSON payload. If later interceptors rewrote the
    visible content, fall back to ``tool_call_args`` so a loaded skill is not
    dropped from the sandbox.
    """
    loaded_order: list[str] = []
    to_remove: set[str] = set()
    for msg in conversation:
        if msg.role != MessageRole.TOOL:
            continue
        call_name = msg.additional_kwargs.get("tool_call_name")
        if call_name not in _SKILL_TOOL_CALL_NAMES:
            continue
        parsed = _parse_tool_result_content(msg.content)
        args = _tool_call_args(msg)
        data = parsed if parsed is not None else args
        skill_name = data.get("name") or args.get("name")
        if not skill_name:
            continue
        if call_name == SKILL_LOAD_TOOL_NAME and "error" not in data:
            if skill_name in loaded_order:
                loaded_order.remove(skill_name)
            loaded_order.append(skill_name)
            to_remove.discard(skill_name)
            if (
                maximum_loaded_skills is not None
                and len(loaded_order) > maximum_loaded_skills
            ):
                evicted = loaded_order.pop(0)
                to_remove.add(evicted)
        elif call_name == SKILL_UNLOAD_TOOL_NAME and (
            (parsed is not None and data.get("unloaded")) or parsed is None
        ):
            if skill_name in loaded_order:
                loaded_order.remove(skill_name)
            to_remove.add(skill_name)
    return set(loaded_order), to_remove


def _resolve_active_skill_names(
    conversation: list[ChatMessage],
    maximum_loaded_skills: int | None = None,
) -> set[str]:
    active, _ = _resolve_skill_states(conversation, maximum_loaded_skills)
    return active


@singleton
class SkillsInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(
        self,
        skill_service: SkillService,
        skill_loader: SkillLoader,
        settings: Settings,
    ) -> None:
        self._skill_service = skill_service
        self._skill_loader = skill_loader
        self._skill_injection_mode = settings.skills.skill_injection_mode

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        filter_input = self._find_skill_filter(state.input.request.tool_context)

        stack = state.input.context_stack
        stack = stack.remove_layers_of_type(LayerType.SKILL_CATALOG)
        stack = stack.remove_layers_of_type(LayerType.SKILL_BODY)
        stack = stack.remove_layers_of_type(LayerType.MOUNTS)
        if filter_input is None:
            state.input.context_stack = stack
            context.set_state(state)
            return

        skills_cache = state.runtime.cache.skill
        entries = skills_cache.entries if skills_cache else []
        resources = skills_cache.resources if skills_cache else {}
        if not entries:
            state.input.context_stack = stack
            context.set_state(state)
            return

        active_skill_names, _ = _resolve_skill_states(
            state.input.request.messages,
            maximum_loaded_skills=state.input.request.context.maximum_loaded_skills,
        )
        active_versions: list[SkillVersionEntity] = []
        mounted_versions: list[SkillVersionEntity] = []
        catalog_entries: list[SkillCatalogEntry] = []
        eager_version_ids: set[str] = set()

        for entry in entries:
            skill = entry.skill
            version = entry.version
            name = version.frontmatter.name
            loading = skill.loading
            if loading == "eager":
                active_versions.append(version)
                eager_version_ids.add(version.id)
            elif name in active_skill_names:
                if self._skill_injection_mode == "system_prompt":
                    active_versions.append(version)
                mounted_versions.append(version)
            else:
                catalog_entries.append(
                    SkillCatalogEntry(
                        id=version.skill_id,
                        name=name,
                        description=version.frontmatter.description,
                        loading=loading,
                    )
                )

        if catalog_entries and self._skill_injection_mode == "system_prompt":
            stack = stack.append_layer(
                SkillCatalogLayer(entries=catalog_entries, source="skills")
            )

        mounted_names = {v.frontmatter.name for v in mounted_versions}

        for version in active_versions:
            try:
                instructions = await self._skill_service.get_skill_body(version)
            except Exception as exc:
                logger.warning(
                    "Skills: unable to load body for %s: %s", version.skill_id, exc
                )
                continue
            is_mounted = version.frontmatter.name in mounted_names
            is_eager = version.id in eager_version_ids
            stack = stack.append_layer(
                SkillBodyLayer(
                    skill_id=version.skill_id,
                    name=version.frontmatter.name,
                    version=version.version,
                    instructions=instructions,
                    location=skill_mount_path(version.frontmatter.name)
                    if is_mounted
                    else "",
                    resources=resources.get(version.skill_id, []) if is_mounted else [],
                    render_as_xml=not is_eager,
                    source=f"skill:{version.frontmatter.name}",
                )
            )

        mounts = self._skill_loader.mounts_for_versions(mounted_versions)
        if mounts:
            stack = stack.append_layer(MountsLayer(mounts=mounts, source="skills"))

        state.input.context_stack = stack
        context.set_state(state)

    def _find_skill_filter(
        self, tool_context: Sequence[ArtifactType]
    ) -> SkillFilter | None:
        for artifact in tool_context:
            if isinstance(artifact, SkillArtifact):
                return artifact.skill_filter
        return None
