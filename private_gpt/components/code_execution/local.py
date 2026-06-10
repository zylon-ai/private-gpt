from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from injector import inject

from private_gpt.components.code_execution.base import (
    CodeExecutionProvider,
)
from private_gpt.components.code_execution.bash_executor import LocalBashExecutor
from private_gpt.components.code_execution.sandbox_session import (
    SandboxCodeExecutionSession,
)
from private_gpt.components.code_execution.workspace_manager import (
    LocalWorkspaceManager,
)
from private_gpt.components.sandbox.local import BashExecutorSandbox
from private_gpt.components.sandbox.mount import (
    LocalMount,
    LocalMountSpec,
    ReadOnlyMount,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import CodeExecutionSession
    from private_gpt.components.code_execution.content_bundle import ContentBundle
    from private_gpt.components.code_execution.workspace_manager import WorkspaceManager
    from private_gpt.components.sandbox.mount import SessionMount


def _cache_key(canonical_path: str) -> str:
    return hashlib.sha1(canonical_path.encode()).hexdigest()[:16]


class LocalCodeExecutionProvider(CodeExecutionProvider):
    """Provides SandboxCodeExecutionSession instances backed by the local filesystem.

    Uses BashExecutorSandbox as the execution backend. Session directories survive
    kernel restarts — existing files are reused on reconnect.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        base = Path(
            settings.code_execution.workspace_path
            or Path(settings.data.local_data_folder) / "code_execution_workspaces"
        )
        self._content_cache = Path(base / "content_cache")
        self._workspace = self._make_workspace_manager(base / "sessions")
        self._workspace.ensure_mounted()
        self._sessions_dir = self._workspace.sessions_path
        self._executor = LocalBashExecutor(
            cpu_limit_seconds=settings.bash.cpu_limit_seconds,
            memory_limit_mb=settings.bash.memory_limit_mb,
            fsize_limit_mb=settings.bash.fsize_limit_mb,
            nproc_limit=settings.bash.nproc_limit,
            output_cap_bytes=settings.bash.output_cap_bytes,
        )
        self._ttl = settings.code_execution.session_ttl_seconds
        self._active: dict[str, tuple[SandboxCodeExecutionSession, list[float]]] = {}
        self._lock = asyncio.Lock()
        self._reaper_started = False

    def _make_workspace_manager(self, base: Path) -> WorkspaceManager:
        """Factory hook — subclasses override to inject cloud-backed storage."""
        return LocalWorkspaceManager(base)

    async def create_session(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> SandboxCodeExecutionSession:
        host_paths = self._workspace.prepare_session_dirs(session_id)

        # Build LocalMountSpec list — typed pydantic models
        local_specs: list[LocalMountSpec] = [
            LocalMountSpec(
                canonical=c, real_path=p, writable=(c != "/mnt/user-data/uploads/")
            )
            for c, p in host_paths.items()
        ]
        for bundle in extra_bundles or []:
            cache_dir = self._content_cache / _cache_key(bundle.canonical_path)
            local_specs.append(
                LocalMountSpec(
                    canonical=bundle.canonical_path,
                    real_path=cache_dir,
                    writable=bundle.writable,
                )
            )

        sandbox = BashExecutorSandbox(local_specs, self._executor)

        # Build SessionMount list — delegates setup to sandbox APIs
        mounts: list[SessionMount] = [LocalMount(spec) for spec in local_specs[:3]]
        for bundle, spec in zip(extra_bundles or [], local_specs[3:], strict=True):
            mounts.append(ReadOnlyMount(spec, bundle.files))

        await asyncio.gather(*[m.prepare(sandbox) for m in mounts])

        last_accessed: list[float] = [time.monotonic()]
        session = SandboxCodeExecutionSession(
            session_id, sandbox, "/home/agent/", last_accessed
        )

        async with self._lock:
            self._active[session_id] = (session, last_accessed)

        self._ensure_reaper()
        return session

    def delete_session(self, session: CodeExecutionSession) -> None:
        if isinstance(session, SandboxCodeExecutionSession):
            sid = session._id
            self._active.pop(sid, None)

    def _ensure_reaper(self) -> None:
        if not self._reaper_started:
            self._reaper_started = True
            asyncio.get_event_loop().create_task(self._reaper_loop())

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            async with self._lock:
                expired = [
                    sid
                    for sid, (_, last) in self._active.items()
                    if now - last[0] > self._ttl
                ]
            for sid in expired:
                async with self._lock:
                    self._active.pop(sid, None)
