from __future__ import annotations

import json
from typing import Any, cast

import redis.asyncio as redis  # type: ignore[import-untyped]
from injector import inject, singleton

from private_gpt.arq.settings import get_redis_settings
from private_gpt.components.engines.chat.checkpoint_store import (
    ChatCheckpoint,
    ChatCheckpointStore,
)
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.settings.settings import Settings


@singleton
class RedisChatCheckpointStore(ChatCheckpointStore):
    """Redis-backed checkpoint storage for multi-process ARQ execution."""

    @inject
    def __init__(self, settings: Settings) -> None:
        redis_settings = get_redis_settings(settings)
        self._redis = redis.Redis(
            host=cast(str, redis_settings.host),
            port=redis_settings.port,
            db=redis_settings.database,
            username=redis_settings.username,
            password=redis_settings.password,
            decode_responses=True,
        )
        self._prefix = "private_gpt:arq:iteration"
        self._ttl = settings.scheduler.chat.callback_timeout_seconds + 300

    def _ctx_key(self, execution_id: str) -> str:
        return f"{self._prefix}:ctx:{execution_id}"

    def _results_key(self, execution_id: str) -> str:
        return f"{self._prefix}:results:{execution_id}"

    def _resumed_key(self, execution_id: str) -> str:
        return f"{self._prefix}:resumed:{execution_id}"

    async def save(self, checkpoint: ChatCheckpoint) -> None:
        await self._redis.set(
            self._ctx_key(checkpoint.correlation_id),
            checkpoint.model_dump_json(),
            ex=self._ttl,
        )
        await self._redis.delete(self._resumed_key(checkpoint.correlation_id))

    async def load(self, execution_id: str) -> ChatCheckpoint | None:
        data = await self._redis.get(self._ctx_key(execution_id))
        return ChatCheckpoint.model_validate_json(data) if data else None

    async def record_result(
        self, execution_id: str, tool_id: str, result: dict[str, Any]
    ) -> dict[str, ToolExecutionResponse] | None:
        checkpoint = await self.load(execution_id)
        results_key = self._results_key(execution_id)
        await cast(Any, self._redis.hset(results_key, tool_id, json.dumps(result)))
        await cast(Any, self._redis.expire(results_key, self._ttl))
        if checkpoint is None:
            return None
        expected = set(checkpoint.checkpoint_payload.pending_async_tools)
        if tool_id not in expected:
            await cast(Any, self._redis.hdel(results_key, tool_id))
            return None
        results = await self.get_results(execution_id)
        return results if expected.issubset(results) else None

    async def get_results(self, execution_id: str) -> dict[str, ToolExecutionResponse]:
        payload = cast(
            dict[str, str],
            await cast(Any, self._redis.hgetall(self._results_key(execution_id))),
        )
        return {
            key: ToolExecutionResponse.model_validate_json(value)
            for key, value in payload.items()
        }

    async def claim_resume(self, execution_id: str) -> bool:
        return bool(
            await self._redis.set(
                self._resumed_key(execution_id), "1", ex=self._ttl, nx=True
            )
        )

    async def cleanup(self, execution_id: str) -> None:
        await self._redis.delete(
            self._ctx_key(execution_id),
            self._results_key(execution_id),
            self._resumed_key(execution_id),
        )

    async def close(self) -> None:
        await self._redis.aclose()
