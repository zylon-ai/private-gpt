import base64
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from injector import singleton
from llama_index.core.base.llms.types import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    TextBlock,
)

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

# Model-facing function-name contract: at most 64 chars from `[A-Za-z0-9_-]`.
MAX_PUBLIC_NAME_LENGTH = 64
INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
# Hex chars of the SHA-256 identity hash appended on lossy normalization.
HASH_LENGTH = 12


def normalize_mcp_tool_name(server_name: str, raw_name: str) -> str:
    """Derive the model-facing public name for one MCP tool.

    Deterministic pure function of ``(server_name, raw_name)``. The clean case
    is ``mcp__<server_name>__<raw_name>`` verbatim. When character replacement
    or truncation to the function-name contract (64 chars, ``[A-Za-z0-9_-]``)
    changes the name, a 12-hex-char SHA-256 hash of the identity is appended so
    distinct MCP identities never collapse into the same public name.

    The raw name is never parsed back out of the public name — calls always
    resolve through ``McpToolDefinition.raw_name``.
    """
    joined = f"mcp__{server_name}__{raw_name}"
    normalized = INVALID_NAME_CHARS.sub("_", joined)
    if normalized == joined and len(normalized) <= MAX_PUBLIC_NAME_LENGTH:
        return normalized
    digest = hashlib.sha256(f"{server_name}\0{raw_name}".encode("utf-8")).hexdigest()
    suffix = digest[:HASH_LENGTH]
    return f"{normalized[: MAX_PUBLIC_NAME_LENGTH - HASH_LENGTH - 1]}_{suffix}"


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
    """Typed MCP tool definition with the original input schema preserved.

    ``name`` is the model-facing public name (collision-safe normalized);
    ``raw_name`` is the MCP server's own tool name, used on the wire.
    """

    name: str
    description: str | None
    input_schema: dict[str, Any]
    raw_name: str | None = None

    @classmethod
    def from_mcp_tool(
        cls, tool: "Tool", server_name: str | None = None
    ) -> "McpToolDefinition":
        schema = tool.input_schema
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        public_name = (
            normalize_mcp_tool_name(server_name, tool.name)
            if server_name
            else tool.name
        )
        return cls(
            name=public_name,
            description=tool.description,
            # Preserve original schema as-is (including untyped properties).
            input_schema=dict(schema),
            raw_name=tool.name,
        )


class McpClient:
    """MCP client that preserves original tool schemas and invokes tools directly."""

    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self.client: PersistentMCPClient | None = None
        self._last_tools: list[McpToolDefinition] = []

        persistent_mcp_client_cls = _load_runtime()

        headers: dict[str, str] = {}
        if config.authorization_token:
            headers["Authorization"] = f"Bearer {config.authorization_token}"

        self.client = persistent_mcp_client_cls(
            command_or_url=config.url,
            headers=headers,
            timeout=10 * 60,
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
            tools.append(
                McpToolDefinition.from_mcp_tool(remote_tool, server_name=self.config.name)
            )
        # Remember the last-listed generation so call_tool can resolve a
        # public name back to the raw MCP name before hitting the wire.
        self._last_tools = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> "CallToolResult":
        """Invoke a tool on the MCP server with the given arguments.

        ``name`` is the model-facing public name; it is resolved to the raw
        MCP tool name before calling, because the wire protocol only
        understands the server's own names.
        """
        if self.client is None:
            raise RuntimeError("MCP client is not initialized")
        raw_name = self._resolve_raw_name(name)
        return await self.client.call_tool(name=raw_name, arguments=arguments)

    def _resolve_raw_name(self, public_name: str) -> str:
        """Map a public name back to the raw MCP tool name, if known.

        Falls back to passing the public name through unchanged so callers
        that pre-date normalization (or that invoke a tool never listed)
        still reach the wire with a best-effort name.
        """
        for tool in getattr(self, "_last_tools", None) or []:
            if tool.name == public_name:
                return tool.raw_name or public_name
        return public_name

    async def close(self) -> None:
        """Close the underlying MCP session in the task that owns it."""
        if self.client is not None:
            await self.client.close()


@singleton
class McpService:
    def create_client(self, config: McpServerConfig) -> McpClient:
        """Create a new MCP client with the given configuration."""
        return McpClient(config)


def mcp_tool_to_spec(config: McpServerConfig, tool: McpToolDefinition) -> "ToolSpec":
    """Convert a discovered MCP tool into a durable, rebuildable tool spec.

    The model-facing ``ToolSpec.name`` is the collision-safe public name; the
    raw MCP name travels in ``tool_name`` so ``call_tool`` hits the wire with
    the server's own name.
    """
    return rebuild_mcp_tool(
        config=config,
        tool_name=tool.raw_name or tool.name,
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
        client = McpClient(config)
        try:
            tools = await client.list_tools()
            # ``name`` is the model-facing public name; the client resolves it
            # back to the raw MCP name before calling the wire protocol.
            public_name = name or tool_name
            available = {item.name for item in tools}
            if public_name not in available:
                raise ValueError(
                    f"MCP tool {public_name!r} is no longer available from "
                    f"server {config.name!r}."
                )
            return await client.call_tool(public_name, dict(kwargs))
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
