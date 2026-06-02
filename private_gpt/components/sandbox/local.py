import os
import subprocess
import sys
import time

from private_gpt.components.sandbox.base import (
    SandboxCodeOptions,
    SandboxExecOptions,
    SandboxExecutionResult,
    SandboxProvider,
    SandboxSession,
)
from private_gpt.settings.settings import Settings


class LocalSandboxSession(SandboxSession):
    """Local sandbox implementation for development and OSS defaults."""

    def __init__(self, timeout: int = 60) -> None:
        self._timeout = timeout
        self._started = False

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    def exec(
        self, command: str, options: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        options = options or SandboxExecOptions()
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=options.cwd,
                env={**os.environ, **options.env} if options.env else None,
                text=True,
                capture_output=True,
                timeout=options.timeout or self._timeout,
                check=False,
            )
            return SandboxExecutionResult(
                success=completed.returncode == 0,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                execution_time_ms=int((time.monotonic() - started_at) * 1000),
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            return SandboxExecutionResult(
                success=False,
                stdout=stdout or "",
                stderr=stderr or f"Command timed out after {e.timeout} seconds.",
                exit_code=124,
                execution_time_ms=int((time.monotonic() - started_at) * 1000),
            )

    def run_code(
        self, code: str, options: SandboxCodeOptions | None = None
    ) -> SandboxExecutionResult:
        options = options or SandboxCodeOptions()
        language = options.language.lower()
        if language in {"bash", "sh", "shell"}:
            return self.exec(code, options)
        return self.exec(_command_for_language(language, code), options)


class LocalSandboxProvider(SandboxProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._timeout = settings.sandbox.timeout

    def create_session(
        self, user_id: str | None = None, timeout: int | None = None
    ) -> SandboxSession:
        return LocalSandboxSession(timeout=timeout or self._timeout)

    def delete_session(self, session: SandboxSession) -> None:
        if isinstance(session, LocalSandboxSession):
            session.close()
        else:
            raise TypeError("Expected a LocalSandboxSession instance.")


def _command_for_language(language: str, code: str) -> str:
    match language:
        case "python" | "py":
            return f"{sys.executable} <<'EOF'\n{code}\nEOF"
        case "javascript" | "js" | "node" | "typescript" | "ts":
            return f"node <<'EOF'\n{code}\nEOF"
        case _:
            return f"{language} <<'EOF'\n{code}\nEOF"
