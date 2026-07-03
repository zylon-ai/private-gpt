from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from private_gpt.components.sandbox.content_bundle import ContentBundle
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.results import (
        BashExecutionResult,
        FileOperationResult,
    )


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
    async def read_file(self, path: str) -> bytes:
        """Read raw file bytes from the session workspace."""

    @abstractmethod
    async def close(self) -> None:
        """Close and release the backing execution session."""


class CodeExecutionSessionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    session_id: str
    extra_bundles: list[ContentBundle] = []
    bundles_to_remove: list[str] = []
    env: dict[str, str] = {}


class CodeExecutionProvider(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def create_session(
        self,
        config: CodeExecutionSessionConfig,
    ) -> CodeExecutionSession:
        """Create a code execution session, optionally mounting extra bundles."""

    @abstractmethod
    def delete_session(self, session: CodeExecutionSession) -> None:
        """Delete a code execution session."""


CodeExecutionProviderFactory = (
    type[CodeExecutionProvider] | Callable[[Settings], CodeExecutionProvider]
)
