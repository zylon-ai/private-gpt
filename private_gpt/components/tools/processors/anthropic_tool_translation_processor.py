import logging

from injector import singleton

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ToolSpec,
)
from private_gpt.components.tools.anthropic_tools import (
    is_anthropic_server_tool_type,
    resolve_anthropic_client_tool,
    resolve_anthropic_server_tool_to_internal,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _replace_tool,
    _wrapper_tool,
)

logger = logging.getLogger(__name__)


@singleton
class AnthropicToolTranslationProcessor(ToolProcessor):
    """Translate Anthropic date-versioned tool type strings to PrivateGPT tool specs.

    Must run first in the pipeline. For each tool matching the Anthropic date-suffix
    pattern (e.g. web_search_20250305, bash_20250124):

    - Server tools with a known translation (web_search_*, web_fetch_*,
      code_execution_*): replaced with an unresolved internal ToolSpec so
      downstream processors resolve them.
    - Client tools (bash_*, text_editor_*, computer_*, memory_*): enriched
      once with the canonical description and input_schema while preserving
      their Anthropic type. The API caller executes these tools.
    - Unknown date-suffix types: discarded with a warning.
    """

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not is_anthropic_server_tool_type(tool.type):
                continue

            # 1. Server tool → translate to internal (PGPT executes)
            internal_name = resolve_anthropic_server_tool_to_internal(tool.type)
            if internal_name is not None:
                translated = _wrapper_tool(
                    name=internal_name,
                )
                _replace_tool(request, tool, [translated])
                return True

            # 2. Client tool → pass-through with canonical schema (caller executes)
            client_spec = resolve_anthropic_client_tool(tool.type)
            if client_spec is not None:
                pass_through = ToolSpec(
                    name=tool.name or client_spec.name,
                    type=tool.type,
                    description=tool.description or client_spec.description,
                    input_schema=tool.input_schema or client_spec.input_schema,
                )
                if pass_through == tool:
                    continue
                _replace_tool(request, tool, [pass_through])
                return True

            # 3. Unknown type → discard
            logger.warning(
                "Discarding tool '%s' (type '%s'): no internal implementation available.",
                tool.name,
                tool.type,
            )
            _replace_tool(request, tool, [])
            return True

        return False
