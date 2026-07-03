from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BashExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: int = 0


class FileOperationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    output: str = ""
    error: str | None = None
