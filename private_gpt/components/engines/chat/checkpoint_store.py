from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, cast

from injector import Injector, inject, singleton
from pydantic import BaseModel, Field

from private_gpt.components.engines.chat.async_chat_engine import (
    IterationCheckpointPayload,
)
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.settings.settings import Settings


class ChatCheckpoint(BaseModel):
    correlation_id: str
    request_data: dict[str, Any]
    stream_type: str
    metadata: dict[str, Any]
    iteration: int
    checkpoint: str = "before_iteration"
    checkpoint_payload: IterationCheckpointPayload = Field(
        default_factory=IterationCheckpointPayload
    )
    next_block_count: int = 0
    checkpoint_id: str = ""
    deadline: datetime | None = None


class ChatCheckpointStore(ABC):
    @abstractmethod
    async def save(self, checkpoint: ChatCheckpoint) -> None:
        ...

    @abstractmethod
    async def load(self, execution_id: str) -> ChatCheckpoint | None:
        ...

    @abstractmethod
    async def record_result(
        self, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> dict[str, ToolExecutionResponse] | None:
        ...

    @abstractmethod
    async def get_results(self, execution_id: str) -> dict[str, ToolExecutionResponse]:
        ...

    @abstractmethod
    async def claim_resume(self, execution_id: str) -> bool:
        ...

    @abstractmethod
    async def cleanup(self, execution_id: str) -> None:
        ...


@singleton
class InMemoryChatCheckpointStore(ChatCheckpointStore):
    """Single-process checkpoint storage for local execution and tests."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, ChatCheckpoint] = {}
        self._results: dict[str, dict[str, ToolExecutionResponse]] = {}
        self._resumed: set[str] = set()
        self._lock = asyncio.Lock()

    async def save(self, checkpoint: ChatCheckpoint) -> None:
        async with self._lock:
            self._checkpoints[checkpoint.correlation_id] = checkpoint
            self._resumed.discard(checkpoint.correlation_id)

    async def load(self, execution_id: str) -> ChatCheckpoint | None:
        async with self._lock:
            return self._checkpoints.get(execution_id)

    async def record_result(
        self, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> dict[str, ToolExecutionResponse] | None:
        async with self._lock:
            checkpoint = self._checkpoints.get(execution_id)
            if checkpoint is None:
                results = self._results.setdefault(execution_id, {})
                results[tool_id] = ToolExecutionResponse.model_validate(result)
                return None
            if tool_id not in checkpoint.checkpoint_payload.pending_async_tools:
                return None
            results = self._results.setdefault(execution_id, {})
            results[tool_id] = ToolExecutionResponse.model_validate(result)
            expected = set(checkpoint.checkpoint_payload.pending_async_tools)
            return dict(results) if expected.issubset(results) else None

    async def get_results(self, execution_id: str) -> dict[str, ToolExecutionResponse]:
        async with self._lock:
            return dict(self._results.get(execution_id, {}))

    async def claim_resume(self, execution_id: str) -> bool:
        async with self._lock:
            if execution_id in self._resumed:
                return False
            self._resumed.add(execution_id)
            return True

    async def cleanup(self, execution_id: str) -> None:
        async with self._lock:
            self._checkpoints.pop(execution_id, None)
            self._results.pop(execution_id, None)
            self._resumed.discard(execution_id)


@singleton
class ChatCheckpointStoreFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector

    def get(self) -> ChatCheckpointStore:
        if self._settings.scheduler.chat.mode == "arq":
            from private_gpt.arq.chat.iteration_state import RedisChatCheckpointStore

            return cast(
                ChatCheckpointStore,
                self._injector.get(RedisChatCheckpointStore),
            )
        return cast(
            ChatCheckpointStore,
            self._injector.get(InMemoryChatCheckpointStore),
        )
