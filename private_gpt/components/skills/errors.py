"""Typed errors for the skills component."""

from enum import StrEnum


class SkillErrorCode(StrEnum):
    """Stable error codes for skill validation failures."""

    # Upload / file-level
    MISSING_FILES = "MISSING_FILES"
    MISSING_SKILL_MD = "MISSING_SKILL_MD"
    INVALID_ZIP = "INVALID_ZIP"
    EMPTY_ZIP = "EMPTY_ZIP"
    UNSAFE_PATH_ABSOLUTE = "UNSAFE_PATH_ABSOLUTE"
    UNSAFE_PATH_TRAVERSAL = "UNSAFE_PATH_TRAVERSAL"
    MIME_TYPE_UNKNOWN = "MIME_TYPE_UNKNOWN"

    # Bundle-level
    BUNDLE_TOO_LARGE = "BUNDLE_TOO_LARGE"

    # Parse / frontmatter
    MISSING_FRONTMATTER = "MISSING_FRONTMATTER"
    INVALID_FRONTMATTER = "INVALID_FRONTMATTER"

    # Frontmatter field validation
    NAME_REQUIRED = "NAME_REQUIRED"
    NAME_TOO_LONG = "NAME_TOO_LONG"
    NAME_INVALID_FORMAT = "NAME_INVALID_FORMAT"
    NAME_CONSECUTIVE_HYPHENS = "NAME_CONSECUTIVE_HYPHENS"
    DESCRIPTION_REQUIRED = "DESCRIPTION_REQUIRED"
    DESCRIPTION_TOO_LONG = "DESCRIPTION_TOO_LONG"
    METADATA_EMPTY_KEY = "METADATA_EMPTY_KEY"
    METADATA_INVALID_VALUE = "METADATA_INVALID_VALUE"


class SkillDomainError(Exception):
    """Typed skill domain error with stable code and human-readable message."""

    def __init__(
        self,
        code: SkillErrorCode,
        message: str,
        params: dict[str, str] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.params = params
        super().__init__(message)


class SkillValidationErrors(Exception):
    """Aggregates multiple SkillDomainErrors from a single validation pass."""

    def __init__(self, errors: list[SkillDomainError]) -> None:
        self.errors = errors
        super().__init__("; ".join(e.message for e in errors))
