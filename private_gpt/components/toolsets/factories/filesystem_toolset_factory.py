"""Provide canonical filesystem toolset factory."""

from uuid import uuid4

from private_gpt.components.toolsets.models.tool_definition import ToolDefinition
from private_gpt.components.toolsets.models.tool_set import ToolSet


def build_filesystem_toolset() -> ToolSet:
    """Build the canonical filesystem toolset definition."""
    return ToolSet(
        id=uuid4(),
        name="filesystem",
        version="1.0.0",
        description="Session-scoped virtual filesystem operations",
        tools=[
            ToolDefinition(
                name="read_file",
                type="read_file_v1",
                description="Read the UTF-8 text content of a file from the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                annotations={"read_only": True, "destructive": False},
            ),
            ToolDefinition(
                name="write_file",
                type="write_file_v1",
                description="Write UTF-8 text content to a file in the virtual filesystem. Pass create_parents=true to create missing parent directories.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file",
                        },
                        "content": {
                            "type": "string",
                            "description": "UTF-8 text to write",
                        },
                        "create_parents": {
                            "type": "boolean",
                            "description": "Create missing parent directories",
                            "default": False,
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                annotations={"read_only": False, "destructive": False},
            ),
            ToolDefinition(
                name="delete",
                type="delete_v1",
                description="Delete a file or directory (recursively) from the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to delete",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                annotations={"read_only": False, "destructive": True},
            ),
            ToolDefinition(
                name="move",
                type="move_v1",
                description="Move (rename) a file or directory inside the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source path"},
                        "dst": {"type": "string", "description": "Destination path"},
                    },
                    "required": ["src", "dst"],
                    "additionalProperties": False,
                },
                annotations={"read_only": False, "destructive": False},
            ),
            ToolDefinition(
                name="copy",
                type="copy_v1",
                description="Copy a file to a new path inside the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source file path"},
                        "dst": {
                            "type": "string",
                            "description": "Destination file path",
                        },
                    },
                    "required": ["src", "dst"],
                    "additionalProperties": False,
                },
                annotations={"read_only": False, "destructive": False},
            ),
            ToolDefinition(
                name="list_dir",
                type="list_dir_v1",
                description="List immediate children of a directory in the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list",
                            "default": "/",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                annotations={"read_only": True, "destructive": False},
            ),
            ToolDefinition(
                name="create_dir",
                type="create_dir_v1",
                description="Create a directory (and all intermediate parents) in the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to create",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                annotations={"read_only": False, "destructive": False},
            ),
            ToolDefinition(
                name="exists",
                type="exists_v1",
                description="Check whether a path exists in the virtual filesystem.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to check"}
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                annotations={"read_only": True, "destructive": False},
            ),
        ],
    )
