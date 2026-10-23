"""Define toolset group model."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from private_gpt.components.toolsets.models.tool_definition import ToolDefinition


class ToolSet(BaseModel):
    """Represent a named group of tool definitions."""

    id: UUID
    name: str
    version: str
    description: str
    tools: list[ToolDefinition]

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_unique_tool_names(self) -> "ToolSet":
        """Validate that tool names are unique in the toolset."""
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique within a toolset")
        return self
