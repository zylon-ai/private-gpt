import re
from typing import Annotated, Literal

from llama_index.core.base.llms.types import TextBlock
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.sandbox.content_bundle import ContentBundle


class BaseContextLayer(BaseModel):
    """Common fields shared by every context layer."""

    source: str = Field(
        default="request",
        description="Origin of the layer, e.g. 'platform', 'skill:git', 'mcp'.",
    )
    priority: int = Field(
        default=1000,
        description=(
            "Render priority for prompt layers. Lower values are rendered first. "
            "State-only layers ignore this field."
        ),
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def render(self) -> str:
        """Return text to include in the system prompt (empty for state layers)."""
        return ""


class UserInstructionsLayer(BaseContextLayer):
    """User/system provided baseline instructions."""

    type: Literal[LayerType.USER_INSTRUCTIONS] = Field(
        default=LayerType.USER_INSTRUCTIONS, frozen=True
    )
    priority: int = Field(default=100, frozen=True)
    text: str | list[TextBlock] | None = Field(description="Raw instruction text.")

    def render(self) -> str:
        if self.text is None:
            return ""
        if isinstance(self.text, str):
            return self.text.strip()

        texts = [block.text.strip() for block in self.text if block.text.strip()]
        return "\n\n".join(texts)


class RuntimeInstructionsLayer(BaseContextLayer):
    """Transient runtime instructions (e.g. condensation hints)."""

    type: Literal[LayerType.RUNTIME_INSTRUCTIONS] = Field(
        default=LayerType.RUNTIME_INSTRUCTIONS, frozen=True
    )
    priority: int = Field(default=200, frozen=True)
    text: str = Field(description="Additional instruction text.")

    def render(self) -> str:
        return self.text


class SkillCatalogEntry(BaseModel):
    id: str = Field(description="Skill identifier.")
    name: str = Field(description="Skill frontmatter name.")
    description: str = Field(description="Skill frontmatter description.")
    loading: Literal["eager", "lazy"] = Field(description="Skill loading mode.")
    location: str = Field(
        default="",
        description="Path to the skill's SKILL.md inside the execution "
        "environment, e.g. /mnt/skills/pdf/SKILL.md.",
    )
    resources: list[str] = Field(
        default_factory=list,
        description="Bundled file paths relative to the skill directory.",
    )


class SkillCatalogLayer(BaseContextLayer):
    """Catalog of available-but-not-yet-loaded skills shown to the LLM."""

    type: Literal[LayerType.SKILL_CATALOG] = Field(
        default=LayerType.SKILL_CATALOG, frozen=True
    )
    priority: int = Field(default=300, frozen=True)
    entries: list[SkillCatalogEntry] = Field(
        default_factory=list,
        description="List of available skill entries.",
    )

    def render(self) -> str:
        if not self.entries:
            return ""
        lines = ["<available_skills>"]
        for entry in self.entries:
            lines.append("  <skill>")
            lines.append(f"    <name>{entry.name}</name>")
            lines.append(f"    <description>{entry.description}</description>")
            if entry.location:
                lines.append(f"    <location>{entry.location}</location>")
            if entry.resources:
                lines.append("    <resources>")
                lines.extend(
                    f"      <resource>{resource}</resource>"
                    for resource in entry.resources
                )
                lines.append("    </resources>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)


class SkillBodyLayer(BaseContextLayer):
    """Full instructions for one activated skill."""

    type: Literal[LayerType.SKILL_BODY] = Field(
        default=LayerType.SKILL_BODY, frozen=True
    )
    priority: int = Field(default=400, frozen=True)
    skill_id: str = Field(description="Skill identifier.")
    name: str = Field(description="Skill frontmatter name.")
    version: str = Field(description="Skill version token.")
    instructions: str = Field(description="Skill instruction body content.")
    location: str = Field(
        default="",
        description="Skill directory inside the execution environment, "
        "e.g. /mnt/skills/pdf/.",
    )
    resources: list[str] = Field(
        default_factory=list,
        description="Bundled file paths relative to the skill directory.",
    )

    def render(self) -> str:
        body = re.sub(r"^#{1,6}\s+.*$", "", self.instructions, flags=re.MULTILINE)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if self.location:
            footer = f"Skill directory: {self.location}"
            if self.resources:
                footer += " (resources: " + ", ".join(self.resources) + ")"
            body = f"{body}\n\n{footer}" if body else footer
        return body


class ToolInstructionsLayer(BaseContextLayer):
    """Per-tool instructions injected when a tool is available."""

    type: Literal[LayerType.TOOL_INSTRUCTIONS] = Field(
        default=LayerType.TOOL_INSTRUCTIONS, frozen=True
    )
    priority: int = Field(default=450, frozen=True)
    tool_name: str = Field(
        description="Canonical tool name these instructions apply to."
    )
    instructions: str = Field(description="Instruction text for this tool.")

    def render(self) -> str:
        return self.instructions


class DocumentLayer(BaseContextLayer):
    """One document injected as context — wraps the real Document entity."""

    type: Literal[LayerType.DOCUMENT] = Field(default=LayerType.DOCUMENT, frozen=True)
    priority: int = Field(default=2000, frozen=True)
    document: Document = Field(description="The Document domain object.")

    def render(self) -> str:
        return ""  # state layer — never in system prompt


class ToolDefinitionsLayer(BaseContextLayer):
    """Tool specs available to the LLM — consumed programmatically, not rendered."""

    type: Literal[LayerType.TOOL_DEFINITIONS] = Field(
        default=LayerType.TOOL_DEFINITIONS, frozen=True
    )
    priority: int = Field(default=2000, frozen=True)
    tools: list[ToolSpec] = Field(
        default_factory=list,
        description="List of ToolSpec instances.",
    )

    def render(self) -> str:
        return ""  # state layer — never in system prompt


class ContextPromptLayer(BaseContextLayer):
    """Rendered context section to include in the system prompt."""

    type: Literal[LayerType.CONTEXT] = Field(default=LayerType.CONTEXT, frozen=True)
    priority: int = Field(default=600, frozen=True)
    text: str = Field(description="Rendered context prompt text.")

    def render(self) -> str:
        return self.text


class ContentBundlesLayer(BaseContextLayer):
    """Skill content bundles — consumed by tool builders, not rendered."""

    type: Literal[LayerType.CONTENT_BUNDLES] = Field(
        default=LayerType.CONTENT_BUNDLES, frozen=True
    )
    priority: int = Field(default=2000, frozen=True)
    bundles: list[ContentBundle] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def render(self) -> str:
        return ""  # state layer — never in system prompt


AnyContextLayer = Annotated[
    UserInstructionsLayer
    | RuntimeInstructionsLayer
    | ContextPromptLayer
    | SkillCatalogLayer
    | SkillBodyLayer
    | ToolInstructionsLayer
    | DocumentLayer
    | ToolDefinitionsLayer
    | ContentBundlesLayer,
    Field(discriminator="type"),
]
