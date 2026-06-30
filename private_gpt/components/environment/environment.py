from __future__ import annotations

import logging
import shlex
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
    from private_gpt.components.sandbox.content_bundle import ContentBundle

logger = logging.getLogger(__name__)


@dataclass
class Environment:
    """A live, mounted sandbox bound to a session id.

    Tools (code execution, bash, ...) share one Environment per session.
    Delegated calls refresh the idle clock the manager's reaper watches, so
    any tool activity keeps the environment alive.

    ContentBundles (skills, tools, ...) are registered lazily via add_pending()
    and materialized just before the first exec() that follows — no network
    calls at registration time. The _stale flag is set on any flush failure so
    the EnvironmentManager can evict and recreate on the next acquire().
    """

    id: str
    sandbox: SandboxSession
    workspace: str
    content_mounters: list[ContentMounter]
    last_accessed: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._mounted: set[str] = set()
        self._pending: list[ContentBundle] = []
        self._stale: bool = False

    def touch(self) -> None:
        self.last_accessed = time.monotonic()

    def idle_seconds(self, now: float) -> float:
        return now - self.last_accessed

    def add_pending(self, bundles: list[ContentBundle]) -> None:
        """Register bundles to be materialized before the next exec().

        Bundles already confirmed materialized in this process lifetime are
        skipped — no duplicate work. Zero network calls.
        """
        for bundle in bundles:
            if bundle.canonical_path not in self._mounted:
                self._pending.append(bundle)

    async def _flush_pending(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        try:
            for bundle in pending:
                mounter = next(
                    (m for m in self.content_mounters if m.can_handle(bundle)), None
                )
                if mounter:
                    await mounter.materialize(bundle, self.sandbox)
                self._mounted.add(bundle.canonical_path)
        except Exception:
            self._stale = True
            raise

    async def remove_bundles(self, canonical_paths: list[str]) -> None:
        for path in canonical_paths:
            normalized = path.rstrip("/")
            await self.sandbox.exec(f"rm -rf {shlex.quote(normalized)}")
            self._mounted.discard(path)

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
