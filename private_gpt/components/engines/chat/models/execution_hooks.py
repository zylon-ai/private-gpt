from typing import Any

from pydantic import BaseModel, Field


class ToolExecutionHook(BaseModel):
    callable_path: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ExecutionHooks(BaseModel):
    tool_result: list[ToolExecutionHook] = Field(default_factory=list)
