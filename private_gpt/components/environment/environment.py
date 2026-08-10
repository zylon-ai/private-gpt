from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from private_gpt.components.sandbox.base import (
        SandboxCodeOptions,
        SandboxExecOptions,
        SandboxExecutionResult,
        SandboxSession,
    )

logger = logging.getLogger(__name__)

# Minimum seconds between two shared-activity writes (Redis round trips are
# fire-and-forget; throttling avoids task churn on hot paths).
_ACTIVITY_THROTTLE_SECONDS = 5.0


@dataclass
class Environment:
    """A live, mounted sandbox bound to a session id.

    Tools (code execution, bash, ...) share one Environment per session.
    Delegated calls refresh the idle clock the manager's reaper watches, so
    any tool activity keeps the environment alive.

    All mounts are bind volumes wired at sandbox creation; nothing is
    materialized into the running container afterwards. When the mount set
    changes, the EnvironmentManager recreates the sandbox.
    """

    id: str
    sandbox: SandboxSession
    workspace: str
    last_accessed: float = field(default_factory=time.monotonic)
    ttl_start: float = field(default_factory=time.monotonic)
    last_renewed: float = field(default_factory=lambda: 0.0)
    # Cross-process ownership marker (the manager's coordinator instance id).
    owner: str = ""
    # Optional async callback used to publish activity to the shared
    # last-activity clock (set by the EnvironmentManager when a distributed
    # coordinator is available).
    activity_sink: Callable[[str], Coroutine[None, None, None]] | None = None

    def __post_init__(self) -> None:
        # Mount fingerprint this env was created with; used by the manager
        # to detect mount changes on reuse.
        self._mount_keys: frozenset[tuple[object, ...]] = frozenset()
        self._sandbox_env: dict[str, str] = {}
        self._last_shared_touch: float = 0.0

    def touch(self) -> None:
        now = time.monotonic()
        self.last_accessed = now
        if (
            self.activity_sink is not None
            and now - self._last_shared_touch >= _ACTIVITY_THROTTLE_SECONDS
        ):
            self._last_shared_touch = now
            with suppress(RuntimeError):
                # Best-effort, fire-and-forget: the shared clock is used by
                # the reaper on OTHER pods/workers.
                asyncio.get_running_loop().create_task(self.activity_sink(self.id))

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
