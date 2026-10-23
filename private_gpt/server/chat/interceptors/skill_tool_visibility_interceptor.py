import re

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import (
    ToolDefinitionsLayer,
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
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_MANAGEMENT_TOOLS,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    _resolve_active_skill_names,
    find_skill_filter,
    partition_skill_entries,
)


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

    async def intercept(self, context: ChatInterceptorContext) -> None:
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
        loaded_names: list[str] = []

        skill_filter = find_skill_filter(state.input.request.tool_context, tools)
        if skill_filter is not None and state.runtime.cache.skill is not None:
            cache = state.runtime.cache.skill
            active_names = _resolve_active_skill_names(state.input.request.messages)
            partition = partition_skill_entries(cache.entries, active_names)
            loaded_names = sorted(partition.loaded_names)
            has_loaded_skill = bool(partition.loaded_names)
            has_catalog = bool(partition.catalog_entries)
            has_loaded_non_eager = bool(partition.active_lazy_versions)
            for version in (
                *partition.eager_versions,
                *partition.active_lazy_versions,
            ):
                if version.frontmatter.allowed_tools:
                    allowed_tools.update(
                        self._normalize_token(item)
                        for item in version.frontmatter.allowed_tools
                    )

        filtered = [
            self._with_loaded_names(tool, loaded_names)
            if tool.name == SKILL_LIST_TOOL_NAME
            else tool
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

    @staticmethod
    def _with_loaded_names(tool: ToolSpec, loaded_names: list[str]) -> ToolSpec:
        metadata = tool.execution_metadata
        if metadata is None:
            return tool
        kwargs = dict(metadata.rebuild_kwargs)
        if kwargs.get("loaded_names") == loaded_names:
            return tool
        kwargs["loaded_names"] = loaded_names
        return tool.model_copy(
            update={
                "execution_metadata": metadata.model_copy(
                    update={"rebuild_kwargs": kwargs}
                )
            }
        )

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

        # Non-deferred tools are explicitly requested and always visible.
        if not tool.defer_loading:
            return True

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
