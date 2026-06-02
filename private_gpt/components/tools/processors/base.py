import uuid
from abc import ABC, abstractmethod

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ToolSpec,
    _dummy_tool_async_fn,
)
from private_gpt.components.tools.tool_names import resolve_internal_tool_name
from private_gpt.server.utils.artifact_input import ArtifactType


class ToolProcessor(ABC):
    """Interceptors may edit the request tool list in place."""

    @abstractmethod
    async def intercept(self, request: ResolvedChatRequest) -> bool:
        """Return True when the request was modified."""


def _get_tool_context(
    request: ResolvedChatRequest,
    tool: ToolSpec,
) -> list[ArtifactType]:
    if tool.context is not None:
        return tool.context
    return request.tool_context or []


def _session_id(request: ResolvedChatRequest) -> str:
    return (
        request.context.user_id or request.context.correlation_id or str(uuid.uuid4())
    )


def _tool_matches(tool: ToolSpec, *tool_names: str) -> bool:
    # Only match on the versioned type (e.g. semantic_search_v1 → semantic_search).
    # Name-based matching would cause external tools that share a name with an internal
    # tool to be incorrectly resolved by internal processors.
    resolved_type = resolve_internal_tool_name(tool.type)
    return resolved_type is not None and resolved_type in tool_names


def _is_unresolved_tool(tool: ToolSpec) -> bool:
    return tool.async_fn is _dummy_tool_async_fn


def _wrapper_tool(
    name: str,
    description: str | None = None,
    tool_type: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or None,
        type=tool_type or f"{name}_v1",
    )


def _replace_tool(
    request: ResolvedChatRequest,
    original: ToolSpec,
    replacements: list[ToolSpec],
) -> bool:
    tools = request.tool_config.tools
    for index, candidate in enumerate(tools):
        if candidate is original:
            request.tool_config.tools = [
                *tools[:index],
                *replacements,
                *tools[index + 1 :],
            ]
            return True
    return False
