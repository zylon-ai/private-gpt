from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from private_gpt.settings.settings import Settings


class SandboxExecutionResult(BaseModel):
    """Result from a sandbox command or code execution."""

    model_config = ConfigDict(frozen=True)

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: int = 0

    @property
    def output(self) -> str:
        return self.stdout

    @property
    def error(self) -> str | None:
        return self.stderr or None

    @property
    def failed(self) -> bool:
        return not self.success


class SandboxExecOptions(BaseModel):
    timeout: int | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None


class SandboxCodeOptions(SandboxExecOptions):
    language: str = Field(
        default="python",
        description="Runtime language identifier, for example python, node, or bash.",
    )


class SandboxSession(ABC):
    @abstractmethod
    def start(self) -> None:
        """Start or attach to the remote sandbox session."""

    @abstractmethod
    def close(self) -> None:
        """Release the remote sandbox session."""

    @abstractmethod
    def exec(
        self, command: str, options: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    def run_code(
        self, code: str, options: SandboxCodeOptions | None = None
    ) -> SandboxExecutionResult:
        """Execute code in a named runtime."""

    def install_package(self, package_name: str) -> SandboxExecutionResult:
        return self.exec(f"python -m pip install {package_name}")

    def __enter__(self) -> "SandboxSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class SandboxProvider(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def create_session(
        self, user_id: str | None = None, timeout: int | None = None
    ) -> SandboxSession:
        """Create a sandbox session. The session may be lazy until start()."""

    @abstractmethod
    def delete_session(self, session: SandboxSession) -> None:
        """Delete a sandbox session and release all associated resources."""


SandboxProviderFactory = type[SandboxProvider] | Callable[[Settings], SandboxProvider]
