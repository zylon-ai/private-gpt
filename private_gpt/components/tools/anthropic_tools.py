import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from private_gpt.components.tools.tool_names import (
    CODE_EXECUTION_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
)

# Anthropic server tool translation
# Keys are glob patterns matching Anthropic's date-versioned type strings.
# Values are the equivalent PrivateGPT internal tool names.
_ANTHROPIC_DATE_SUFFIX_RE = re.compile(r"_\d{8}$")

ANTHROPIC_SERVER_TOOL_TRANSLATION: dict[str, str] = {
    "web_search_*": WEB_SEARCH_TOOL_NAME,
    "web_fetch_*": WEB_FETCH_TOOL_NAME,
    "code_execution_*": CODE_EXECUTION_TOOL_NAME,
}


def is_anthropic_server_tool_type(tool_type: str | None) -> bool:
    """Return True for Anthropic date-versioned type strings (e.g. web_search_20250305).

    Matches any type string ending in an 8-digit date suffix (_YYYYMMDD).
    """
    return bool(tool_type and _ANTHROPIC_DATE_SUFFIX_RE.search(tool_type))


def resolve_anthropic_server_tool_to_internal(tool_type: str | None) -> str | None:
    """Return the internal tool name for a server tool type, or None if unknown."""
    if not tool_type:
        return None
    for pattern, internal_name in ANTHROPIC_SERVER_TOOL_TRANSLATION.items():
        if fnmatch.fnmatch(tool_type, pattern):
            return internal_name
    return None


# Anthropic client tool specs
# These are tools the API caller executes. PrivateGPT provides description +
# input_schema so the model knows how to invoke them, but does not run them locally.
@dataclass(frozen=True)
class _AnthropicClientToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


ANTHROPIC_CLIENT_TOOL_TRANSLATION: dict[str, _AnthropicClientToolSpec] = {
    "bash_*": _AnthropicClientToolSpec(
        name="bash",
        description="Execute bash commands in a persistent shell session.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "restart": {
                    "type": "boolean",
                    "description": "Restart the shell session before running the command.",
                },
            },
        },
    ),
    "text_editor_*": _AnthropicClientToolSpec(
        name="str_replace_based_edit_tool",
        description="View and edit files using string replacement operations.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "str_replace", "create", "insert"],
                    "description": "The operation to perform.",
                },
                "path": {"type": "string", "description": "Absolute path to the file."},
                # view
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[start_line, end_line] to view (view only).",
                },
                # str_replace
                "old_str": {
                    "type": "string",
                    "description": "Exact text to replace (str_replace).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text (str_replace).",
                },
                # create
                "file_text": {
                    "type": "string",
                    "description": "Full file content (create).",
                },
                # insert
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert after; 0 = beginning (insert).",
                },
                "insert_text": {
                    "type": "string",
                    "description": "Text to insert (insert).",
                },
            },
            "required": ["command", "path"],
        },
    ),
    "computer_*": _AnthropicClientToolSpec(
        name="computer",
        description="Control a computer via mouse, keyboard, and screenshot actions.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "cursor_position",
                        "left_click",
                        "right_click",
                        "double_click",
                        "mouse_move",
                        "scroll",
                        "type",
                        "key",
                    ],
                    "description": "The action to perform.",
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[x, y] screen coordinate.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type or key to press.",
                },
            },
            "required": ["action"],
        },
    ),
    "memory_*": _AnthropicClientToolSpec(
        name="memory",
        description="Manage a persistent memory file store across conversation turns.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "view",
                        "create",
                        "str_replace",
                        "insert",
                        "delete",
                        "rename",
                    ],
                    "description": "The memory operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Path within the memory store.",
                },
                # create
                "file_text": {
                    "type": "string",
                    "description": "File content (create).",
                },
                # str_replace
                "old_str": {
                    "type": "string",
                    "description": "Exact text to replace (str_replace).",
                },
                # str_replace / insert
                "new_str": {
                    "type": "string",
                    "description": "Replacement or inserted text (str_replace, insert).",
                },
                # insert
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert after (insert).",
                },
                # rename
                "new_path": {"type": "string", "description": "New path (rename)."},
            },
            "required": ["command", "path"],
        },
    ),
}


def resolve_anthropic_client_tool(
    tool_type: str | None,
) -> _AnthropicClientToolSpec | None:
    """Return the client tool spec for a client tool type, or None if unknown."""
    if not tool_type:
        return None
    for pattern, spec in ANTHROPIC_CLIENT_TOOL_TRANSLATION.items():
        if fnmatch.fnmatch(tool_type, pattern):
            return spec
    return None


async def _client_tool_placeholder_async_fn(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "This tool is executed by the API caller. PrivateGPT does not run it locally."
    )
