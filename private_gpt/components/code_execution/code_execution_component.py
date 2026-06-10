from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.code_execution.registry import CodeExecutionProviderRegistry
from private_gpt.settings.settings import Settings  # noqa: TC001

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionProvider,
        CodeExecutionSession,
    )
    from private_gpt.components.code_execution.content_bundle import ContentBundle


@singleton
class CodeExecutionComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = CodeExecutionProviderRegistry(settings)
        self._sessions: dict[str, CodeExecutionSession] = {}
        self._lock = asyncio.Lock()

    def _get_code_execution_provider(self) -> CodeExecutionProvider | None:
        provider_name = self._settings.code_execution.provider
        return self._registry.get_provider(provider_name) if provider_name else None

    async def get_or_create_session(
        self,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> CodeExecutionSession | None:
        provider = self._get_code_execution_provider()
        if not provider:
            return None

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                return session

            session = await provider.create_session(
                session_id,
                extra_bundles=extra_bundles,
            )
            self._sessions[session_id] = session
            return session

    async def delete_session(self, session_id: str) -> None:
        provider = self._get_code_execution_provider()
        if not provider:
            return None

        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                provider.delete_session(session)
