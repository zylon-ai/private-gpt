"""Define tool definition model and ToolSpec conversion."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from private_gpt.components.chat.models.chat_config_models import ToolSpec

_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ToolDefinition(BaseModel):
    """Represent one callable tool schema exposed to an LLM."""

    name: str
    type: str | None = None
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate the tool name pattern."""
        if not _TOOL_NAME_PATTERN.fullmatch(value):
            raise ValueError("Tool name must match [a-zA-Z0-9_-]+")
        return value

    def to_tool_spec(self) -> ToolSpec:
        """Convert this tool definition into the project ToolSpec model."""
        return ToolSpec(
            name=self.name,
            type=self.type,
            description=self.description,
            input_schema=self.input_schema,
        )
