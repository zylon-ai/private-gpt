import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

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
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillVersionEntity,
    SkillVersionWithSkillEntity,
)
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_loader import SkillLoader
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.server.utils.artifact_input import ArtifactType, SkillArtifact
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_SKILL_TOOL_CALL_NAMES = {SKILL_LOAD_TOOL_NAME, SKILL_UNLOAD_TOOL_NAME}


def _parse_tool_result_content(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str) or not content.strip():
        return None
    raw = content.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _tool_call_args(msg: ChatMessage) -> dict[str, Any]:
    args = msg.additional_kwargs.get("tool_call_args")
    return args if isinstance(args, dict) else {}


def _message_text(msg: ChatMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return str(content or "")


def _iter_tool_selections(msg: ChatMessage) -> list[tuple[str, dict[str, Any], str]]:
    """Return (tool_name, kwargs, tool_id) for assistant tool_calls."""
    raw = msg.additional_kwargs.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    selections: list[tuple[str, dict[str, Any], str]] = []
    for item in raw:
        if isinstance(item, ToolSelection):
            kwargs = item.tool_kwargs if isinstance(item.tool_kwargs, dict) else {}
            selections.append((item.tool_name, kwargs, item.tool_id or ""))
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name") or item.get("name")
        if not name and isinstance(item.get("function"), dict):
            name = item["function"].get("name")
        kwargs = item.get("tool_kwargs") or item.get("arguments") or {}
        if isinstance(kwargs, str):
            parsed = _parse_tool_result_content(kwargs)
            kwargs = parsed or {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        tool_id = str(item.get("tool_id") or item.get("id") or "")
        if isinstance(name, str):
            selections.append((name, kwargs, tool_id))
    return selections


def _apply_skill_event(
    *,
    call_name: str,
    data: dict[str, Any],
    args: dict[str, Any],
    parsed: dict[str, Any] | None,
    loaded_order: list[str],
    to_remove: set[str],
    maximum_loaded_skills: int | None,
) -> None:
    skill_name = data.get("name") or args.get("name")
    if not skill_name:
        return
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
    elif (call_name == SKILL_LOAD_TOOL_NAME and "error" in data) or (
        call_name == SKILL_UNLOAD_TOOL_NAME
        and ((parsed is not None and data.get("unloaded")) or parsed is None)
    ):
        if skill_name in loaded_order:
            loaded_order.remove(skill_name)
        to_remove.add(skill_name)


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
    dropped from the sandbox. Client follow-ups sometimes keep the
    ``load_skill`` assistant tool call but drop the TOOL observation; treat
    that call as a successful load unless a later result reports an error.
    """
    loaded_order: list[str] = []
    to_remove: set[str] = set()
    for msg in conversation:
        if msg.role == MessageRole.ASSISTANT:
            for call_name, kwargs, _tool_id in _iter_tool_selections(msg):
                if call_name not in _SKILL_TOOL_CALL_NAMES:
                    continue
                _apply_skill_event(
                    call_name=call_name,
                    data=kwargs,
                    args=kwargs,
                    parsed=None,
                    loaded_order=loaded_order,
                    to_remove=to_remove,
                    maximum_loaded_skills=maximum_loaded_skills,
                )

        parsed = _parse_tool_result_content(_message_text(msg))
        args = _tool_call_args(msg)
        call_name = msg.additional_kwargs.get("tool_call_name")
        if call_name not in _SKILL_TOOL_CALL_NAMES:
            if parsed and parsed.get("loaded") is True and parsed.get("name"):
                call_name = SKILL_LOAD_TOOL_NAME
            elif parsed and parsed.get("unloaded") is True and parsed.get("name"):
                call_name = SKILL_UNLOAD_TOOL_NAME
            else:
                continue
        if msg.role not in {
            MessageRole.TOOL,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        }:
            continue
        data = parsed if parsed is not None else args
        _apply_skill_event(
            call_name=call_name,
            data=data,
            args=args,
            parsed=parsed,
            loaded_order=loaded_order,
            to_remove=to_remove,
            maximum_loaded_skills=maximum_loaded_skills,
        )
    return set(loaded_order), to_remove


def _resolve_active_skill_names(
    conversation: list[ChatMessage],
    maximum_loaded_skills: int | None = None,
) -> set[str]:
    active, _ = _resolve_skill_states(conversation, maximum_loaded_skills)
    return active


def find_skill_filter(
    tool_context: Sequence[ArtifactType] | None = None,
    tools: Sequence[Any] | None = None,
) -> SkillFilter | None:
    """Resolve the skill filter from request context or per-tool context."""
    for artifact in tool_context or []:
        if isinstance(artifact, SkillArtifact):
            return artifact.skill_filter
    for tool in tools or []:
        for artifact in getattr(tool, "context", None) or []:
            if isinstance(artifact, SkillArtifact):
                return artifact.skill_filter
    return None


@dataclass(frozen=True)
class _SkillPartition:
    eager_versions: list[SkillVersionEntity]
    active_lazy_versions: list[SkillVersionEntity]
    catalog_entries: list[SkillCatalogEntry]
    eager_version_ids: set[str]
    loaded_names: set[str]


def partition_skill_entries(
    entries: Sequence[SkillVersionWithSkillEntity],
    active_names: set[str],
) -> _SkillPartition:
    """Split resolved skills into always-loaded eager, active lazy, and catalog.

    Eager skills are loaded for every turn (body + mounts). Lazy skills are
    loaded only after ``load_skill`` (``active_names``). Everything else is
    catalog-only — the set ``list_skills`` should return.
    """
    eager_versions: list[SkillVersionEntity] = []
    active_lazy_versions: list[SkillVersionEntity] = []
    catalog_entries: list[SkillCatalogEntry] = []
    eager_version_ids: set[str] = set()
    loaded_names: set[str] = set()

    for entry in entries:
        version = entry.version
        name = version.frontmatter.name
        if entry.skill.loading == "eager":
            eager_versions.append(version)
            eager_version_ids.add(version.id)
            loaded_names.add(name)
        elif name in active_names:
            active_lazy_versions.append(version)
            loaded_names.add(name)
        else:
            catalog_entries.append(
                SkillCatalogEntry(
                    id=version.skill_id,
                    name=name,
                    description=version.frontmatter.description,
                    loading=entry.skill.loading,
                )
            )

    return _SkillPartition(
        eager_versions=eager_versions,
        active_lazy_versions=active_lazy_versions,
        catalog_entries=catalog_entries,
        eager_version_ids=eager_version_ids,
        loaded_names=loaded_names,
    )


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
        stack = state.input.context_stack
        filter_input = find_skill_filter(
            state.input.request.tool_context,
            stack.all_tools(),
        )

        stack = stack.remove_layers_of_type(LayerType.SKILL_CATALOG)
        stack = stack.remove_layers_of_type(LayerType.SKILL_BODY)
        stack = stack.remove_layers_of_source("skills")
        if filter_input is None:
            # No skill context on this request: catalog/body are not applicable.
            state.input.context_stack = stack
            context.set_state(state)
            return

        skills_cache = state.runtime.cache.skill
        entries = skills_cache.entries if skills_cache else []
        resources = skills_cache.resources if skills_cache else {}
        if not entries:
            # Cache is the input used to rebuild. Empty means no skills this
            # iteration — do not freeze a previous catalog/body, those depend
            # on the current messages (load/unload).
            state.input.context_stack = stack
            context.set_state(state)
            return

        active_skill_names, _ = _resolve_skill_states(
            state.input.request.messages,
            maximum_loaded_skills=state.input.request.context.maximum_loaded_skills,
        )
        partition = partition_skill_entries(entries, active_skill_names)
        body_versions: list[SkillVersionEntity] = list(partition.eager_versions)
        mounted_versions: list[SkillVersionEntity] = [
            *partition.eager_versions,
            *partition.active_lazy_versions,
        ]
        if self._skill_injection_mode == "system_prompt":
            body_versions.extend(partition.active_lazy_versions)

        if partition.catalog_entries and self._skill_injection_mode == "system_prompt":
            stack = stack.append_layer(
                SkillCatalogLayer(entries=partition.catalog_entries, source="skills")
            )

        mounted_names = {v.frontmatter.name for v in mounted_versions}

        for version in body_versions:
            try:
                instructions = await self._skill_service.get_skill_body(version)
            except Exception as exc:
                logger.warning(
                    "Skills: unable to load body for %s: %s", version.skill_id, exc
                )
                continue
            is_mounted = version.frontmatter.name in mounted_names
            is_eager = version.id in partition.eager_version_ids
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
