from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from injector import inject

from private_gpt.components.code_execution.base import (
    CodeExecutionProvider,
)
from private_gpt.components.code_execution.sandbox_session import (
    SandboxCodeExecutionSession,
)
from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.environment.mounter import LocalDirMounter
from private_gpt.components.sandbox.local import LocalSandboxProvider
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import CodeExecutionSession
    from private_gpt.components.environment.mounter import Mounter
    from private_gpt.components.sandbox.content_bundle import ContentBundle


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
        self._manager = EnvironmentManager(
            sandbox_provider=LocalSandboxProvider(settings),
            mounter=self._make_mounter(base),
            ttl_seconds=settings.code_execution.session_ttl_seconds,
        )

    def _make_mounter(self, base: Path) -> Mounter:
        """Factory hook — subclasses override to inject cloud-backed storage."""
        storage_root = (
            Path(self.settings.data.local_data_folder) / "storage"
            if self.settings.skills.storage_provider == "local"
            else None
        )
        return LocalDirMounter(base, storage_root=storage_root)

    async def create_session(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> SandboxCodeExecutionSession:
        env = await self._manager.acquire(session_id, extra_bundles)
        return SandboxCodeExecutionSession(env)

    def delete_session(self, session: CodeExecutionSession) -> None:
        if isinstance(session, SandboxCodeExecutionSession):
            self._manager.release(session._id)
