from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class SkillFrontmatter(BaseModel):
    name: str = Field(
        description="Skill slug name from SKILL.md frontmatter.",
        min_length=1,
        max_length=64,
    )
    description: str = Field(
        description="Human-readable skill usage description from SKILL.md.",
        min_length=1,
        max_length=1024,
    )
    license: str | None = Field(default=None, description="Optional skill license.")
    compatibility: str | None = Field(
        default=None,
        description="Optional environment compatibility constraints.",
        min_length=1,
        max_length=500,
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Optional user-defined key/value metadata from frontmatter.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allowed-tools list parsed from frontmatter.",
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(
        cls, value: dict[str, object] | None
    ) -> dict[str, str] | None:
        if value is None:
            return value
        return {
            key: str(val) if not isinstance(val, str) else val
            for key, val in value.items()
            if key
        }


class SkillEntity(BaseModel):
    id: str = Field(description="Unique skill identifier.")
    collection: str = Field(description="Tenant collection identifier.")
    display_title: str = Field(description="Human display title.")
    source: Literal["custom", "anthropic", "zylon"] = Field(
        description="Skill source provider."
    )
    loading: Literal["eager", "lazy"] = Field(description="Skill loading mode.")
    readonly: bool = Field(description="Readonly flag.")
    latest_version: str | None = Field(
        default=None,
        description="Latest version token derived from versions.",
    )
    created_at: datetime = Field(description="Skill creation timestamp.")
    updated_at: datetime = Field(description="Skill update timestamp.")


class SkillVersionEntity(BaseModel):
    id: str = Field(description="Unique skill version identifier.")
    skill_id: str = Field(description="Parent skill identifier.")
    version: str = Field(description="Version token.")
    frontmatter: SkillFrontmatter = Field(description="Parsed SKILL.md frontmatter.")
    storage_prefix: str = Field(description="Object storage prefix for this version.")
    created_at: datetime = Field(description="Version creation timestamp.")


class SkillVersionWithSkillEntity(BaseModel):
    """Resolved relationship between a version and its parent skill."""

    skill: SkillEntity = Field(description="Parent skill metadata.")
    version: SkillVersionEntity = Field(description="Resolved skill version.")


class SkillReference(BaseModel):
    """Compact pointer to a skill stored in the zylon-gpt skills store.

    Stored by any entity (org, project, future backend artifact) that owns skills.
    """

    skill_id: str = Field(description="Skill identifier in the zylon-gpt skills store.")


class SkillFilter(BaseModel):
    """Collection-scoped filter used to resolve active skills and versions."""

    collection: str = Field(
        description="Tenant collection boundary used to recover skills.",
        min_length=1,
        max_length=255,
    )
    skill_or_version_ids: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("skill_or_version_ids", "skill_ids"),
        serialization_alias="skill_or_version_ids",
        description=(
            "Optional whitelist of identifiers inside collection. "
            "Each item may be a skill id (resolved to latest version) or a skill version id."
        ),
    )


class SkillVersionFile(BaseModel):
    """A file entry inside a skill version bundle (metadata, optionally with content)."""

    path: str = Field(description="Relative path from the skill root.")
    size_bytes: int = Field(description="File size in bytes.")
    mime_type: str | None = Field(
        default=None, description="Detected MIME type, when available."
    )
    content: bytes | None = Field(
        default=None,
        description="Raw file bytes when content was requested.",
    )
