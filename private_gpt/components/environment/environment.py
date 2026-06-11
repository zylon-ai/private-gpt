from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from private_gpt.components.sandbox.base import (
        SandboxCodeOptions,
        SandboxExecOptions,
        SandboxExecutionResult,
        SandboxSession,
    )


@dataclass
class Environment:
    """A live, mounted sandbox bound to a session id.

    Tools (code execution, bash, ...) share one Environment per session.
    Delegated calls refresh the idle clock the manager's reaper watches, so
    any tool activity keeps the environment alive.
    """

    id: str
    sandbox: SandboxSession
    workspace: str  # canonical working directory
    last_accessed: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    def idle_seconds(self, now: float) -> float:
        return now - self.last_accessed

    async def exec(
        self, command: str, opts: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        self.touch()
        return await self.sandbox.exec(command, opts)

    async def run_code(
        self, code: str, opts: SandboxCodeOptions | None = None
    ) -> SandboxExecutionResult:
        self.touch()
        return await self.sandbox.run_code(code, opts)
