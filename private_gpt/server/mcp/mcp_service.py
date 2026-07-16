import base64
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from injector import singleton
from llama_index.core.base.llms.types import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    TextBlock,
)
from llama_index.core.tools import FunctionTool

from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec
    from private_gpt.server.mcp._runtime import (
        AudioContent,
        CallToolResult,
        ImageContent,
        PersistentMCPClient,
        TextContent,
    )


def _load_runtime() -> tuple[
    type["PersistentMCPClient"],
    Callable[..., Awaitable[list[FunctionTool]]],
]:
    try:
        from private_gpt.server.mcp._runtime import (
            PersistentMCPClient,
            aget_tools_from_mcp_url,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message("MCP tools", extras="tool-mcp")
        ) from e

    return PersistentMCPClient, aget_tools_from_mcp_url


def _load_content_types() -> tuple[
    type["TextContent"],
    type["ImageContent"],
    type["AudioContent"],
]:
    from private_gpt.server.mcp._runtime import AudioContent, ImageContent, TextContent

    return TextContent, ImageContent, AudioContent


def _load_call_tool_result_type() -> type["CallToolResult"]:
    from private_gpt.server.mcp._runtime import CallToolResult

    return CallToolResult


def is_mcp_content_block(block: object) -> bool:
    try:
        (
            text_content_type,
            image_content_type,
            audio_content_type,
        ) = _load_content_types()
    except ImportError:
        return False

    return isinstance(
        block, (text_content_type, image_content_type, audio_content_type)
    )


def is_mcp_tool_result(value: object) -> bool:
    try:
        call_tool_result_type = _load_call_tool_result_type()
    except ImportError:
        return False

    return isinstance(value, call_tool_result_type)


def get_mcp_tool_result_content(value: object) -> list[object] | None:
    if not is_mcp_tool_result(value):
        return None

    return cast("CallToolResult", value).content


class McpClient:
    """A simple MCP client that can be used to interact with an MCP server."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self.client: PersistentMCPClient | None = None
        self._aget_tools_from_mcp_url: (
            Callable[..., Awaitable[list[FunctionTool]]] | None
        ) = None

        persistent_mcp_client_cls, aget_tools_from_mcp_url = _load_runtime()

        headers: dict[str, str] = {}
        if config.authorization_token:
            headers["Authorization"] = f"Bearer {config.authorization_token}"

        self.client = persistent_mcp_client_cls(
            command_or_url=config.url,
            headers=headers,
            timeout=10 * 60,
        )
        self._aget_tools_from_mcp_url = aget_tools_from_mcp_url

    async def list_tools(self) -> list[FunctionTool]:
        """Get tools from the MCP server."""
        if self.client is None or self._aget_tools_from_mcp_url is None:
            return []

        if (
            self.config.tool_configuration
            and not self.config.tool_configuration.enabled
        ):
            return []

        tool_list = await self._aget_tools_from_mcp_url(
            command_or_url=self.config.url,
            client=self.client,
        )

        if self.config.tool_configuration.allowed_tools is not None:
            enabled_tools = self.config.tool_configuration.allowed_tools
            if enabled_tools:
                tool_list = [
                    tool for tool in tool_list if tool.metadata.name in enabled_tools
                ]

        return tool_list

    async def close(self) -> None:
        """Close the underlying MCP session in the task that owns it."""
        if self.client is not None:
            await self.client.close()


@singleton
class McpService:
    def create_client(self, config: McpServerConfig) -> McpClient:
        """Create a new MCP client with the given configuration."""
        return McpClient(config)


def mcp_tool_to_spec(config: McpServerConfig, tool: FunctionTool) -> "ToolSpec":
    """Convert a discovered MCP tool into a durable, rebuildable tool spec."""
    from private_gpt.components.chat.models.chat_config_models import ToolSpec

    discovered = ToolSpec.from_llama_index(tool)
    return rebuild_mcp_tool(
        config=config,
        tool_name=tool.metadata.name or discovered.name or "mcp_tool",
        name=discovered.name,
        type=discovered.type,
        description=discovered.description,
        input_schema=discovered.input_schema,
    )


def rebuild_mcp_tool(
    config: McpServerConfig,
    tool_name: str,
    name: str | None,
    type: str | None,
    description: str | None,
    input_schema: dict[str, object] | None,
) -> "ToolSpec":
    """Build an MCP tool whose client session is scoped to each invocation."""
    from private_gpt.components.chat.models.chat_config_models import (
        ToolExecutionMetadata,
        ToolSpec,
    )

    async def invoke_mcp_tool(**kwargs: object) -> object:
        client = McpClient(config)
        try:
            tools = await client.list_tools()
            for tool in tools:
                if tool.metadata.name == tool_name:
                    return await tool.async_fn(**kwargs)
            raise ValueError(
                f"MCP tool {tool_name!r} is no longer available from "
                f"server {config.name!r}."
            )
        finally:
            await client.close()

    rebuild_kwargs = {
        "config": config,
        "tool_name": tool_name,
        "name": name,
        "type": type,
        "description": description,
        "input_schema": input_schema,
    }
    return ToolSpec.from_defaults(
        name=name or tool_name,
        type=type,
        runtime="server",
        description=description,
        input_schema=input_schema,
        async_fn=invoke_mcp_tool,
        execution_metadata=ToolExecutionMetadata(
            rebuild_callable="private_gpt.server.mcp.mcp_service:rebuild_mcp_tool",
            rebuild_kwargs=rebuild_kwargs,
        ),
    )


def convert_mcp_blocks_to_llama_index(block: object) -> ContentBlock | None:
    try:
        (
            text_content_type,
            image_content_type,
            audio_content_type,
        ) = _load_content_types()
    except ImportError:
        return None

    if isinstance(block, text_content_type):
        return TextBlock(text=block.text)
    if isinstance(block, image_content_type):
        bytes_arr = base64.b64decode(block.data)
        return ImageBlock(image=bytes_arr, image_mimetype=block.mimeType)
    if isinstance(block, audio_content_type):
        bytes_arr = base64.b64decode(block.data)
        return AudioBlock(audio=bytes_arr, format=block.mimeType)
    return None
