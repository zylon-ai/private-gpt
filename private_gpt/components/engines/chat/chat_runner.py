from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Protocol, cast

from injector import Injector, inject, singleton

from private_gpt.events.models import Event

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine


class ChatRunner(Protocol):
    async def submit(
        self,
        *,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
        execution_id: str | None = None,
    ) -> tuple[str, AsyncGenerator[Event, None]]: ...

    async def cancel(self, execution_id: str) -> bool: ...

    async def start(
        self,
        *,
        engine: "AsyncChatEngine",
        execution_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None: ...

    async def resume(
        self,
        *,
        engine: "AsyncChatEngine",
        execution_id: str,
        checkpoint_id: str,
    ) -> None: ...

    async def callback(
        self,
        *,
        execution_id: str,
        tool_id: str,
        result: dict[str, Any],
    ) -> None: ...


@singleton
class ChatRunnerFactory:
    @inject
    def __init__(self, injector: Injector) -> None:
        self._injector = injector

    def get(self) -> ChatRunner:
        from private_gpt.components.engines.chat.resumable_runner import (
            ResumableChatRunner,
        )

        return cast(ChatRunner, self._injector.get(ResumableChatRunner))
