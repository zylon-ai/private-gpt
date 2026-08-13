import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from injector import singleton
from llama_index.core.base.llms.types import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    TextBlock,
)

from private_gpt.events.models._events import McpTokensRefreshedEvent
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from mcp.types import (
        AudioContent,
        CallToolResult,
        ImageContent,
        TextContent,
        Tool,
    )

    from private_gpt.components.chat.models.chat_config_models import ToolSpec
    from private_gpt.server.mcp._runtime import PersistentMCPClient


def _load_runtime() -> type["PersistentMCPClient"]:
    try:
        from private_gpt.server.mcp._runtime import PersistentMCPClient
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message("MCP tools", extras="tool-mcp")
        ) from e

    return PersistentMCPClient


def _load_content_types() -> tuple[
    type["TextContent"],
    type["ImageContent"],
    type["AudioContent"],
]:
    from mcp.types import AudioContent, ImageContent, TextContent

    return TextContent, ImageContent, AudioContent


def _load_call_tool_result_type() -> type["CallToolResult"]:
    from mcp.types import CallToolResult

    return CallToolResult


def is_mcp_content_block(block: object) -> bool:
    try:
        text_content_type, image_content_type, audio_content_type = (
            _load_content_types()
        )
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

    result = cast("CallToolResult", value)
    return list(result.content)


@dataclass(frozen=True)
class McpToolDefinition:
    """Typed MCP tool definition with the original input schema preserved."""

    name: str
    description: str | None
    input_schema: dict[str, Any]

    @classmethod
    def from_mcp_tool(cls, tool: "Tool") -> "McpToolDefinition":
        schema = tool.input_schema
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        return cls(
            name=tool.name,
            description=tool.description,
            # Preserve original schema as-is (including untyped properties).
            input_schema=dict(schema),
        )


@dataclass
class McpToolExecutionResult:
    """Internal MCP result that keeps refresh events out of tool content."""

    result: "CallToolResult | None" = None
    error: BaseException | None = None
    events: list[McpTokensRefreshedEvent] = field(default_factory=list)


class McpClient:
    """MCP client that preserves original tool schemas and invokes tools directly."""

    def __init__(
        self,
        config: McpServerConfig,
        emit_event: Callable[[McpTokensRefreshedEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.client: PersistentMCPClient | None = None
        self._emit_event = emit_event

        persistent_mcp_client_cls = _load_runtime()

        headers: dict[str, str] = {}
        if config.authorization_token:
            headers["Authorization"] = f"Bearer {config.authorization_token}"

        self.client = persistent_mcp_client_cls(
            command_or_url=config.url,
            headers=headers,
            timeout=10 * 60,
            refresh_token=config.refresh_token,
            client_id=config.client_id,
            token_endpoint=config.token_endpoint,
            oauth_resource=config.oauth_resource,
            on_tokens_refreshed=self._save_refreshed_tokens,
        )

    def _save_refreshed_tokens(self, access_token: str, refresh_token: str) -> None:
        previous_refresh_token = self.config.refresh_token
        self.config.authorization_token = access_token
        self.config.refresh_token = refresh_token
        if self.config.artifact_id and self._emit_event:
            if previous_refresh_token is None:
                raise RuntimeError("MCP refresh token state is missing")
            self._emit_event(
                McpTokensRefreshedEvent(
                    artifact_id=self.config.artifact_id,
                    previous_refresh_token=previous_refresh_token,
                    authorization_token=access_token,
                    refresh_token=refresh_token,
                )
            )

    async def list_tools(self) -> list[McpToolDefinition]:
        """List tools from the MCP server using the original input schema."""
        if self.client is None:
            return []

        if (
            self.config.tool_configuration
            and not self.config.tool_configuration.enabled
        ):
            return []

        response = await self.client.list_tools()
        allowed = self.config.tool_configuration.allowed_tools

        tools: list[McpToolDefinition] = []
        for remote_tool in response.tools:
            if allowed is not None and remote_tool.name not in allowed:
                continue
            tools.append(McpToolDefinition.from_mcp_tool(remote_tool))
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> "CallToolResult":
        """Invoke a tool on the MCP server with the given arguments."""
        if self.client is None:
            raise RuntimeError("MCP client is not initialized")
        return await self.client.call_tool(name=name, arguments=arguments)

    async def close(self) -> None:
        """Close the underlying MCP session in the task that owns it."""
        if self.client is not None:
            await self.client.close()


@singleton
class McpService:
    def create_client(
        self,
        config: McpServerConfig,
        emit_event: Callable[[McpTokensRefreshedEvent], None] | None = None,
    ) -> McpClient:
        """Create a new MCP client with the given configuration."""
        return McpClient(config, emit_event=emit_event)


def mcp_tool_to_spec(config: McpServerConfig, tool: McpToolDefinition) -> "ToolSpec":
    """Convert a discovered MCP tool into a durable, rebuildable tool spec."""
    return rebuild_mcp_tool(
        config=config,
        tool_name=tool.name,
        name=tool.name,
        type=None,
        description=tool.description,
        input_schema=tool.input_schema,
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
        events: list[McpTokensRefreshedEvent] = []
        client = McpClient(config, emit_event=events.append)
        try:
            try:
                tools = await client.list_tools()
                available = {item.name for item in tools}
                if tool_name not in available:
                    raise ValueError(
                        f"MCP tool {tool_name!r} is no longer available from "
                        f"server {config.name!r}."
                    )
                result = await client.call_tool(tool_name, dict(kwargs))
            except Exception as error:
                if events:
                    return McpToolExecutionResult(error=error, events=events)
                raise
            return McpToolExecutionResult(result=result, events=events)
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
        input_schema=cast(dict[str, Any] | None, input_schema),
        async_fn=invoke_mcp_tool,
        execution_metadata=ToolExecutionMetadata(
            rebuild_callable="private_gpt.server.mcp.mcp_service:rebuild_mcp_tool",
            rebuild_kwargs=rebuild_kwargs,
        ),
    )


def convert_mcp_blocks_to_llama_index(block: object) -> ContentBlock | None:
    try:
        text_content_type, image_content_type, audio_content_type = (
            _load_content_types()
        )
    except ImportError:
        return None

    if isinstance(block, text_content_type):
        text_block = cast("TextContent", block)
        return TextBlock(text=text_block.text)
    if isinstance(block, image_content_type):
        image_block = cast("ImageContent", block)
        bytes_arr = base64.b64decode(image_block.data)
        return ImageBlock(image=bytes_arr, image_mimetype=image_block.mime_type)
    if isinstance(block, audio_content_type):
        audio_block = cast("AudioContent", block)
        bytes_arr = base64.b64decode(audio_block.data)
        return AudioBlock(audio=bytes_arr, format=audio_block.mime_type)
    return None
