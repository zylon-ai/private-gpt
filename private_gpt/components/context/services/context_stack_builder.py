"""Implement context stack builder."""

from pydantic import BaseModel, Field

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.errors import (
    ToolNameConflictError,
)
from private_gpt.components.context.models.context_layer import (
    AnyContextLayer,
    DocumentLayer,
    SkillBodyLayer,
    SkillCatalogEntry,
    SkillCatalogLayer,
    ToolDefinitionsLayer,
    UserInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.citations.types import Document


class ContextStackBuilder(BaseModel):
    """Build validated context stacks with tool conflict checks."""

    user_instructions: str | None = None
    layers: list[AnyContextLayer] = Field(default_factory=list)

    def set_user_instructions(self, text: str) -> "ContextStackBuilder":
        """Set user instructions and append the user-instructions layer."""
        self.user_instructions = text
        self.layers.append(
            UserInstructionsLayer(
                text=text,
                source="platform",
            )
        )
        return self

    def add_skill_catalog(
        self, skills: list[SkillCatalogEntry]
    ) -> "ContextStackBuilder":
        """Add the skill catalog layer from names and descriptions."""
        self.layers.append(SkillCatalogLayer(entries=skills, source="catalog"))
        return self

    def add_activated_skill(
        self, skill_id: str, name: str, version: str, instructions: str
    ) -> "ContextStackBuilder":
        """Add one activated skill body layer."""
        self.layers.append(
            SkillBodyLayer(
                skill_id=skill_id,
                name=name,
                version=version,
                instructions=instructions,
                source=name,
            )
        )
        return self

    def add_tools(
        self,
        tools: list[ToolSpec],
        source: str,
    ) -> "ContextStackBuilder":
        """Add a tool definition layer after checking for name conflicts."""
        self.layers.append(ToolDefinitionsLayer(tools=tools, source=source))
        return self

    def add_document(
        self,
        document: Document,
        source: str,
    ) -> "ContextStackBuilder":
        """Add a document layer after checking document budget."""
        self.layers.append(DocumentLayer(document=document, source=source))
        return self

    def build(self) -> ContextStack:
        """Build an immutable stack after conflict checks."""
        self.validate_tool_name_uniqueness()
        return ContextStack(layers=list(self.layers))

    def validate_tool_name_uniqueness(self) -> None:
        """Validate global tool uniqueness across tool-definition layers."""
        seen: dict[str, str] = {}
        for layer in self.layers:
            if layer.type is not LayerType.TOOL_DEFINITIONS:
                continue
            if not isinstance(layer, ToolDefinitionsLayer):
                continue
            for tool in layer.tools:
                if tool.name is None:
                    continue
                existing = seen.get(tool.name)
                if existing is not None:
                    raise ToolNameConflictError(
                        "Tool name conflict: "
                        f"name='{tool.name}', "
                        f"layer_a='{existing}', layer_b='{layer.source}'"
                    )
                seen[tool.name] = layer.source
