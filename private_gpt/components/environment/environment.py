from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from private_gpt.components.environment.content_mounter import ContentMounter
    from private_gpt.components.sandbox.base import (
        SandboxCodeOptions,
        SandboxExecOptions,
        SandboxExecutionResult,
        SandboxSession,
    )
    from private_gpt.components.sandbox.mount import Mount

logger = logging.getLogger(__name__)


@dataclass
class Environment:
    """A live, mounted sandbox bound to a session id.

    Tools (code execution, bash, ...) share one Environment per session.
    Delegated calls refresh the idle clock the manager's reaper watches, so
    any tool activity keeps the environment alive.

    Mounts (skills, tools, artifacts, ...) are registered via add_pending()
    and materialized by _flush_pending(). When the sandbox is being created
    for the first time, mounts that couldn't be bind-mounted are deferred and
    flushed before the first exec(). When the sandbox is already running,
    the manager flushes immediately so mounts are available right away.
    The _stale flag is set on any flush failure so the EnvironmentManager
    can evict and recreate on the next acquire().
    """

    id: str
    sandbox: SandboxSession
    workspace: str
    content_mounters: list[ContentMounter]
    last_accessed: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._mounted: set[str] = set()
        self._pending: list[Mount] = []
        self._stale: bool = False
        # Mount fingerprint this env was created with. Used by the manager to
        # decide whether a reuse request changed mounts (then it recreates
        # instead of materializing into a running container).
        self._mount_keys: frozenset[tuple[object, ...]] = frozenset()
        self._sandbox_env: dict[str, str] = {}

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    def idle_seconds(self, now: float) -> float:
        return now - self.last_accessed

    def add_pending(self, mounts: list[Mount]) -> None:
        """Stage mounts for materialization, skipping already-mounted targets.

        When the container is already running, the caller is responsible for
        calling _flush_pending() immediately after. When the container is being
        created, deferred mounts are flushed before the first exec().
        """
        for mount in mounts:
            if mount.target not in self._mounted:
                self._pending.append(mount)

    async def _flush_pending(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        try:
            for mount in pending:
                mounter = next(
                    (m for m in self.content_mounters if m.can_handle(mount)), None
                )
                if mounter:
                    await mounter.materialize(mount, self.sandbox)
                self._mounted.add(mount.target)
        except Exception:
            self._stale = True
            raise

    async def remove_mounts(self, targets: list[str]) -> None:
        for target in targets:
            await self.sandbox.remove_mount(target)
            self._mounted.discard(target)

    async def exec(
        self, command: str, opts: SandboxExecOptions | None = None
    ) -> SandboxExecutionResult:
        self.touch()
        await self._flush_pending()
        return await self.sandbox.exec(command, opts)

    async def run_code(
        self, code: str, opts: SandboxCodeOptions | None = None
    ) -> SandboxExecutionResult:
        self.touch()
        await self._flush_pending()
        return await self.sandbox.run_code(code, opts)
