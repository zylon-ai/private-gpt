from typing import Any

from pydantic import BaseModel, Field


class OpenAPIValidationErrorResponse(BaseModel):
    """Standard 422 validation error response."""

    detail: str | list[dict[str, Any]] = Field(
        description="Error detail — a string message or a list of validation error objects.",
    )
