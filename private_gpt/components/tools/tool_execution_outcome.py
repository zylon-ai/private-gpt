from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from private_gpt.events.models import ResultContentBlockType


class ToolExecutionError(BaseModel):
    code: str = "execution_failed"
    message: str
    exception_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionSuccess(BaseModel):
    type: Literal["success"] = "success"
    content: list[ResultContentBlockType] = Field(default_factory=list)


class ToolExecutionFailure(BaseModel):
    type: Literal["failure"] = "failure"
    error: ToolExecutionError


ToolExecutionOutcome = Annotated[
    ToolExecutionSuccess | ToolExecutionFailure,
    Field(discriminator="type"),
]
