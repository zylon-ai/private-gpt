from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from injector import inject

from private_gpt.components.code_execution.base import CodeExecutionProvider
from private_gpt.components.code_execution.sandbox_session import (
    SandboxCodeExecutionSession,
)
from private_gpt.components.environment.hydration import HydratingEnvironmentManager
from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.environment.mounter import LocalDirMounter
from private_gpt.components.sandbox.local import LocalSandboxProvider
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.components.environment.mounter import LayoutMounter


class LocalCodeExecutionProvider(CodeExecutionProvider):
    """Code execution tool over locally managed environments.

    A thin adapter: the EnvironmentManager owns session lifecycle, the
    LocalDirMounter owns the host directories (which survive sandbox
    restarts), and the local sandbox provider owns execution.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        base = Path(
            settings.code_execution.workspace_path
            or Path(settings.data.local_data_folder) / "code_execution_workspaces"
        )
        manager = EnvironmentManager(
            sandbox_provider=LocalSandboxProvider(settings),
            layout_mounter=self._make_layout_mounter(base),
            ttl_seconds=settings.code_execution.session_ttl_seconds,
            namespaces=settings.filesystems.namespaces,
        )
        self._manager = HydratingEnvironmentManager(
            manager=manager, namespaces=settings.filesystems.namespaces
        )

    def _make_layout_mounter(self, base: Path) -> LayoutMounter:
        """Factory hook — subclasses override to inject cloud-backed storage.

        When the 'session' filesystem namespace is configured, sessions are
        rooted there so that files uploaded via the Files API are accessible
        to the sandbox at the same host paths where LocalObjectStorage writes
        them.
        """
        session_ns = self.settings.filesystems.namespaces.get("session")
        if session_ns is not None and session_ns.root:
            return LocalDirMounter(Path(session_ns.root))
        return LocalDirMounter(base)

    async def create_session(
        self,
        config: CodeExecutionSessionConfig,
    ) -> SandboxCodeExecutionSession:
        env = await self._manager.acquire(
            config.session_id,
            mounts=config.mounts or None,
            sandbox_env=config.env or None,
        )
        return SandboxCodeExecutionSession(env)

    def delete_session(self, session: CodeExecutionSession) -> None:
        if isinstance(session, SandboxCodeExecutionSession):
            self._manager.release(session._id)
