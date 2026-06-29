import re
from collections.abc import Sequence

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import (
    ToolDefinitionsLayer,
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
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_MANAGEMENT_TOOLS,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    _resolve_active_skill_names,
)
from private_gpt.server.utils.artifact_input import ArtifactType, SkillArtifact


@singleton
class SkillToolVisibilityInterceptor(ChatRequestLoopInterceptor):
    """Apply skill-aware tool visibility rules.

    - Hide tools with defer_loading=True until at least one skill is loaded.
    - If loaded skills define frontmatter allowed_tools, only expose matching tools
      (supports both internal tools and custom tools by name/type).
    """

    @inject
    def __init__(self) -> None:
        pass

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        stack = state.input.context_stack
        tools = list(stack.all_tools())
        if not tools:
            return

        has_loaded_skill = False
        has_catalog = False  # at least one non-eager skill not yet loaded
        has_loaded_non_eager = False  # at least one non-eager skill currently loaded
        allowed_tools: set[str] = set()

        skill_filter = self._find_skill_filter(state.input.request.tool_context)
        if skill_filter is not None and state.runtime.cache.skill is not None:
            cache = state.runtime.cache.skill
            entries = cache.entries
            active_names = _resolve_active_skill_names(state.input.request.messages)

            for entry in entries:
                skill = entry.skill
                version = entry.version
                loading = skill.loading
                name = version.frontmatter.name
                is_eager = loading == "eager"
                is_active = is_eager or name in active_names

                if is_active:
                    has_loaded_skill = True
                    if not is_eager:
                        has_loaded_non_eager = True
                    if version.frontmatter.allowed_tools:
                        allowed_tools.update(
                            self._normalize_token(item)
                            for item in version.frontmatter.allowed_tools
                        )
                else:
                    has_catalog = True

        filtered = [
            tool
            for tool in tools
            if self._is_visible(
                tool=tool,
                has_loaded_skill=has_loaded_skill,
                has_catalog=has_catalog,
                has_loaded_non_eager=has_loaded_non_eager,
                allowed_tools=allowed_tools,
            )
        ]

        stack = stack.remove_layers_of_type(LayerType.TOOL_DEFINITIONS)
        if filtered:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=filtered, source="skill_tool_visibility")
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

    def _is_visible(
        self,
        tool: ToolSpec,
        has_loaded_skill: bool,
        has_catalog: bool,
        has_loaded_non_eager: bool,
        allowed_tools: set[str],
    ) -> bool:
        tool_name = tool.name or ""

        if tool_name == SKILL_LOAD_TOOL_NAME:
            return has_catalog
        if tool_name == SKILL_UNLOAD_TOOL_NAME:
            return has_loaded_non_eager
        if tool_name == SKILL_LIST_TOOL_NAME:
            return has_catalog

        if tool.defer_loading and not has_loaded_skill:
            return False

        if not allowed_tools:
            return True

        # Once visible, do not further restrict skill-management controls.
        if tool_name in SKILL_MANAGEMENT_TOOLS:
            return True

        return bool(self._tool_tokens(tool) & allowed_tools)

    def _tool_tokens(self, tool: ToolSpec) -> set[str]:
        tokens: set[str] = set()
        if tool.name:
            tokens.add(self._normalize_token(tool.name))
        if tool.type:
            tokens.add(self._normalize_token(tool.type))
            tokens.add(self._normalize_token(re.sub(r"_v\d+$", "", tool.type)))
        return tokens

    @staticmethod
    def _normalize_token(value: str) -> str:
        return value.strip().lower()
