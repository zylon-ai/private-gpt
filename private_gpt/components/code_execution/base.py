from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.content_bundle import ContentBundle


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


class CodeExecutionSession(ABC):
    @abstractmethod
    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        """Execute a bash command in the session workspace."""

    @abstractmethod
    async def view(
        self, path: str, view_range: tuple[int, int] | None = None
    ) -> FileOperationResult:
        """View a file or directory from the session workspace."""

    @abstractmethod
    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        """Replace a single string occurrence in a file."""

    @abstractmethod
    async def create(self, path: str, file_text: str) -> FileOperationResult:
        """Create a new file in the session workspace."""

    @abstractmethod
    async def insert(
        self, path: str, insert_line: int, new_str: str
    ) -> FileOperationResult:
        """Insert text into a file after a given line number."""

    @abstractmethod
    async def close(self) -> None:
        """Close and release the backing execution session."""


class CodeExecutionProvider(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def create_session(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> CodeExecutionSession:
        """Create a code execution session, optionally mounting extra bundles."""

    @abstractmethod
    def delete_session(self, session: CodeExecutionSession) -> None:
        """Delete a code execution session."""


CodeExecutionProviderFactory = (
    type[CodeExecutionProvider] | Callable[[Settings], CodeExecutionProvider]
)
