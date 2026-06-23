import re

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from private_gpt.components.skills.errors import (
    SkillDomainError,
    SkillErrorCode,
    SkillValidationErrors,
)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillFrontmatter(BaseModel):
    name: str = Field(description="Skill slug name", min_length=1, max_length=64)
    description: str = Field(
        description="When and how the skill should be used",
        min_length=1,
        max_length=1024,
    )
    license: str | None = Field(default=None)
    compatibility: str | None = Field(default=None)
    metadata: dict[str, str] | None = Field(default=None)
    allowed_tools_raw: str | None = Field(default=None, alias="allowed-tools")

    @property
    def allowed_tools(self) -> list[str] | None:
        raw = self.allowed_tools_raw
        if raw is None:
            return None
        tools = [token.strip() for token in raw.split(" ") if token.strip()]
        return tools or None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError(
                "name must be lowercase alphanumeric with single hyphens only"
            )
        if "--" in value:
            raise ValueError("name cannot contain consecutive hyphens")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return value
        for key, item in value.items():
            if not key:
                raise ValueError("metadata keys must be non-empty")
            if not isinstance(item, str):
                raise ValueError("metadata values must be strings")
        return value

    @field_validator("allowed_tools_raw", mode="before")
    @classmethod
    def normalize_list_or_str(cls, value: str | list[str] | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            return " ".join(str(item).strip() for item in value if str(item).strip())
        return value


class ParsedSkillDocument(BaseModel):
    frontmatter: SkillFrontmatter
    body: str = Field(default="")


def parse_skill_markdown(skill_markdown: str) -> ParsedSkillDocument:
    match = _FRONTMATTER_RE.match(skill_markdown)
    if not match:
        raise SkillDomainError(
            SkillErrorCode.MISSING_FRONTMATTER,
            "SKILL.md must start with YAML frontmatter",
        )

    raw_frontmatter = match.group(1)
    try:
        parsed_yaml = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as e:
        raise SkillDomainError(
            SkillErrorCode.INVALID_FRONTMATTER,
            "The SKILL.md frontmatter is not valid YAML.",
        ) from e
    if not isinstance(parsed_yaml, dict):
        raise SkillDomainError(
            SkillErrorCode.INVALID_FRONTMATTER,
            "Invalid SKILL.md frontmatter",
        )

    try:
        frontmatter = SkillFrontmatter.model_validate(parsed_yaml)
    except ValidationError as exc:
        errors = [_pydantic_error_to_skill_error(dict(e)) for e in exc.errors()]
        raise SkillValidationErrors(errors) from exc

    body = skill_markdown[match.end() :].strip()
    return ParsedSkillDocument(frontmatter=frontmatter, body=body)


def _pydantic_error_to_skill_error(error: dict[str, object]) -> SkillDomainError:
    loc: tuple[object, ...] = error.get("loc", ())  # type: ignore[assignment]
    error_type = error.get("type", "")
    msg = str(error.get("msg", ""))
    field = str(loc[0]) if loc else ""

    if field == "name":
        if error_type in ("string_too_short", "missing"):
            return SkillDomainError(SkillErrorCode.NAME_REQUIRED, msg)
        if error_type == "string_too_long":
            return SkillDomainError(SkillErrorCode.NAME_TOO_LONG, msg)
        if "consecutive hyphens" in msg:
            return SkillDomainError(SkillErrorCode.NAME_CONSECUTIVE_HYPHENS, msg)
        return SkillDomainError(SkillErrorCode.NAME_INVALID_FORMAT, msg)

    if field == "description":
        if error_type in ("string_too_short", "missing"):
            return SkillDomainError(SkillErrorCode.DESCRIPTION_REQUIRED, msg)
        return SkillDomainError(SkillErrorCode.DESCRIPTION_TOO_LONG, msg)

    if field == "metadata":
        if "keys" in msg:
            return SkillDomainError(SkillErrorCode.METADATA_EMPTY_KEY, msg)
        return SkillDomainError(SkillErrorCode.METADATA_INVALID_VALUE, msg)

    return SkillDomainError(SkillErrorCode.INVALID_FRONTMATTER, msg)
