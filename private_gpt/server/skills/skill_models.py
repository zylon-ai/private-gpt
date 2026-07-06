import base64
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.storage.models import StoredFile


class SkillFileInput(BaseModel):
    path: str = Field(
        description="Relative path from skill root",
        min_length=1,
        max_length=512,
    )
    content_base64: str = Field(description="Base64 content", min_length=1)
    mime_type: str | None = Field(default=None, description="Optional mime type")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if value.startswith("/"):
            raise ValueError("path must be relative and cannot contain '..'")
        if "\\" in value:
            raise ValueError("path must use '/' separators only")
        if ".." in value.split("/"):
            raise ValueError("path must be relative and cannot contain '..'")
        return value

    def to_payload(self) -> StoredFile:
        return StoredFile(
            path=self.path,
            content=base64.b64decode(self.content_base64),
            mime_type=self.mime_type,
        )


class CreateSkillBody(BaseModel):
    display_title: str = Field(
        description="Display title for the skill.",
        min_length=1,
        max_length=255,
    )
    collection: str = Field(
        description="Tenant identifier (org_id boundary).",
        min_length=1,
        max_length=255,
    )
    source: Literal["custom", "anthropic", "zylon"] = Field(
        default="custom",
        description="Source of the skill.",
    )
    loading: Literal["eager", "lazy"] = Field(
        default="lazy",
        description="Instruction loading strategy.",
    )
    readonly: bool = Field(
        default=False, description="Readonly flag for protected skills."
    )
    skill_md: str | None = Field(
        default=None,
        description="Inline SKILL.md content (optional when files includes SKILL.md).",
    )
    files: list[SkillFileInput] = Field(
        default_factory=list,
        description="Optional uploaded files for this skill version.",
    )

    @model_validator(mode="after")
    def validate_skill_md_presence(self) -> "CreateSkillBody":
        has_skill_file = any(file.path == "SKILL.md" for file in self.files)
        if not has_skill_file and not self.skill_md:
            raise ValueError("Provide SKILL.md either in files or skill_md")
        return self


class SkillResponse(BaseModel):
    """Serialized skill object returned by skills endpoints."""

    id: str = Field(description="Unique skill identifier.")
    created_at: datetime = Field(description="Creation timestamp.")
    display_title: str = Field(description="Human display title.")
    latest_version: str | None = Field(
        default=None,
        description="Latest version token for this skill.",
    )
    source: Literal["custom", "anthropic", "zylon"] = Field(
        description="Source of the skill."
    )
    type: Literal["skill"] = Field(default="skill", description="Object type.")
    updated_at: datetime = Field(description="Update timestamp.")
    collection: str = Field(description="Tenant identifier (org_id boundary).")
    loading: Literal["eager", "lazy"] = Field(
        description="Instruction loading strategy."
    )
    readonly: bool = Field(description="Readonly flag.")


class ListSkillsResponse(BaseModel):
    """Paginated list response containing skill objects."""

    data: list[SkillResponse] = Field(
        default_factory=list, description="Skills page data."
    )
    has_more: bool = Field(description="Whether there is a next page.")
    next_page: str | None = Field(default=None, description="Token for next page.")


class SkillDeletedResponse(BaseModel):
    """Deletion marker returned after a skill is removed."""

    id: str = Field(description="Deleted skill identifier.")
    type: Literal["skill_deleted"] = Field(
        default="skill_deleted", description="Deleted object type."
    )


class SkillVersionResponse(BaseModel):
    """Serialized skill version object returned by version endpoints."""

    id: str = Field(description="Unique skill version identifier.")
    created_at: datetime = Field(description="Creation timestamp.")
    description: str = Field(description="Skill version description from frontmatter.")
    directory: str = Field(description="Top-level directory name for this version.")
    name: str = Field(description="Skill name from frontmatter.")
    skill_id: str = Field(description="Parent skill identifier.")
    type: Literal["skill_version"] = Field(
        default="skill_version", description="Object type."
    )
    version: str = Field(description="Version token.")


class ListSkillVersionsResponse(BaseModel):
    """Paginated list response containing skill version objects."""

    data: list[SkillVersionResponse] = Field(
        default_factory=list,
        description="Skill version page data.",
    )
    has_more: bool = Field(description="Whether there is a next page.")
    next_page: str | None = Field(default=None, description="Token for next page.")


class SkillVersionDeletedResponse(BaseModel):
    """Deletion marker returned after a skill version is removed."""

    id: str = Field(description="Deleted version token.")
    type: Literal["skill_version_deleted"] = Field(
        default="skill_version_deleted",
        description="Deleted object type.",
    )


class RecoverSkillsBody(BaseModel):
    skill_filter: SkillFilter = Field(
        description="Filter used to recover active skills by collection and optional whitelist.",
    )


class RecoverSkillsResponse(BaseModel):
    data: list[SkillVersionResponse] = Field(
        default_factory=list,
        description="Recovered skill versions resolved by filter.",
    )


class SkillValidationError(BaseModel):
    """A single structured validation error with stable code and message."""

    code: str = Field(description="Uppercase error code identifying the failure kind.")
    message: str = Field(description="Human-readable description of the error.")
    params: dict[str, str] | None = Field(
        default=None,
        description="Optional i18n interpolation parameters (e.g. {'size': '10'}).",
    )


class SkillValidationResponse(BaseModel):
    """Result of a dry-run skill validation."""

    valid: bool = Field(description="Whether the skill payload would be accepted.")
    name: str | None = Field(
        default=None,
        description="Parsed skill name from SKILL.md frontmatter (when valid).",
    )
    description: str | None = Field(
        default=None,
        description="Parsed skill description from SKILL.md frontmatter (when valid).",
    )
    errors: list[SkillValidationError] = Field(
        default_factory=list,
        description="Validation errors (when invalid).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "valid": True,
                    "name": "sales-ops-helper",
                    "description": "Helps sales reps draft outreach.",
                    "errors": [],
                },
                {
                    "valid": False,
                    "name": None,
                    "description": None,
                    "errors": [
                        {
                            "code": "INVALID_FRONTMATTER",
                            "message": "SKILL.md must start with YAML frontmatter",
                        }
                    ],
                },
            ]
        }
    }
