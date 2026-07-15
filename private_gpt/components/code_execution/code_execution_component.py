from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.code_execution.registry import CodeExecutionProviderRegistry
from private_gpt.components.container_registry import ContainerRegistry
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionProvider,
        CodeExecutionSession,  # noqa: TC004
        CodeExecutionSessionConfig,
    )
    from private_gpt.components.sandbox.base import SandboxLink


@singleton
class CodeExecutionComponent:
    @inject
    def __init__(
        self, settings: Settings, container_registry: ContainerRegistry
    ) -> None:
        self._settings = settings
        self._registry = CodeExecutionProviderRegistry(settings)
        self._container_registry = container_registry
        self._sessions: dict[str, CodeExecutionSession] = {}
        self._lock = asyncio.Lock()

    def _get_code_execution_provider(self) -> CodeExecutionProvider | None:
        provider_name = self._settings.code_execution.provider
        return self._registry.get_provider(provider_name) if provider_name else None

    async def get_or_create_session(
        self,
        config: CodeExecutionSessionConfig,
    ) -> CodeExecutionSession | None:
        provider = self._get_code_execution_provider()
        if not provider:
            return None

        async with self._lock:
            session = await provider.create_session(config)
            self._sessions[config.session_id] = session
            self._container_registry.register(
                config.session_id, self._settings.code_execution.session_ttl_seconds
            )
            return session

    async def get_session_endpoint(
        self, session_id: str, port: int
    ) -> SandboxLink | None:
        from private_gpt.components.code_execution.sandbox_session import (
            SandboxCodeExecutionSession,
        )

        session = self._sessions.get(session_id)
        if not isinstance(session, SandboxCodeExecutionSession):
            return None
        return await session.get_endpoint(port)

    async def delete_session(self, session_id: str) -> None:
        provider = self._get_code_execution_provider()
        if not provider:
            return None

        async with self._lock:
            session = self._sessions.pop(session_id, None)
            self._container_registry.unregister(session_id)
            if session is not None:
                provider.delete_session(session)
