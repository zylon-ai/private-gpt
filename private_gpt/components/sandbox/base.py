from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.sandbox.content_bundle import BundledFile
    from private_gpt.components.sandbox.mount import SandboxMountSpec, VolumeSpec


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
    """Async sandbox session with exec + file operations.

    Permission enforcement: write_file() and chmod() check if the path is
    in a writable mount.
    initialize_mount() bypasses this check — for session setup only.
    make_dir() does not check writable.
    """

    python_executable: str = "python"

    @abstractmethod
    async def exec(
        self, command: str, opts: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        """Execute a shell command."""

    async def run_code(
        self, code: str, opts: SandboxCodeOptions | None = None
    ) -> SandboxExecutionResult:
        """Execute code in a named runtime."""
        opts = opts or SandboxCodeOptions()
        language = opts.language.lower()
        if language in {"bash", "sh", "shell"}:
            return await self.exec(code, opts)
        return await self.exec(self._command_for_language(language, code), opts)

    async def install_package(self, package_name: str) -> SandboxExecutionResult:
        return await self.exec(
            f"{self.python_executable} -m pip install {package_name}"
        )

    def _command_for_language(self, language: str, code: str) -> str:
        match language:
            case "python" | "py":
                return f"{self.python_executable} <<'EOF'\n{code}\nEOF"
            case "javascript" | "js" | "node" | "typescript" | "ts":
                return f"node <<'EOF'\n{code}\nEOF"
            case _:
                return f"{language} <<'EOF'\n{code}\nEOF"

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read file content."""

    @abstractmethod
    async def write_file(self, path: str, content: bytes) -> None:
        """Write file content. Raises ValueError if path is in a read-only mount."""

    @abstractmethod
    async def path_exists(self, path: str) -> bool:
        """Return True if the path exists."""

    @abstractmethod
    async def is_dir(self, path: str) -> bool:
        """Return True if the path is a directory."""

    @abstractmethod
    async def list_dir(self, path: str) -> list[str]:
        """List directory contents as '[dir] name' / '[file] name' strings."""

    @abstractmethod
    async def make_dir(
        self, path: str, *, parents: bool = True, exist_ok: bool = True
    ) -> None:
        """Create directory. Does not check writable."""

    @abstractmethod
    async def chmod(self, path: str, mode: int) -> None:
        """Set file permissions. Raises ValueError if path is in a read-only mount."""

    @abstractmethod
    async def initialize_mount(self, canonical: str, files: list[BundledFile]) -> None:
        """Write mount files during session setup. Bypasses writable check."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""


class SandboxProvider(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def create_session(
        self,
        user_id: str | None = None,
        timeout: int | None = None,
        bundle_specs: list[SandboxMountSpec] | None = None,
        *,
        session_id: str | None = None,
        volumes: list[VolumeSpec] | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxSession:
        """Create a sandbox session. The session may be lazy until first use.

        ``session_id`` tags the backend resource so it can be found again by
        restore_session(); ``volumes`` are host directories to bind-mount.
        ``env`` carries environment variables to inject into the sandbox.
        Backends without those capabilities may ignore them.
        """

    async def restore_session(
        self,
        session_id: str,
        timeout: int | None = None,
        bundle_specs: list[SandboxMountSpec] | None = None,
    ) -> SandboxSession | None:
        """Reattach to an existing backend sandbox for this session, if any.

        Default: the backend cannot restore — returns None.
        """
        return None

    async def renew_session(self, session: SandboxSession) -> None:  # noqa: B027
        """Extend the backend-side lifetime of a live session. Default: no-op."""

    async def kill_session(
        self, session: SandboxSession, session_id: str | None = None
    ) -> None:
        """Forcefully release a session's backend resources.

        Default: delegate to delete_session().
        """
        await self.delete_session(session)

    @abstractmethod
    async def delete_session(self, session: SandboxSession) -> None:
        """Delete a sandbox session and release all associated resources."""


SandboxProviderFactory = type[SandboxProvider] | Callable[[Settings], SandboxProvider]
