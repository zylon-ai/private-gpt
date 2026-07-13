from __future__ import annotations

import json
from typing import Any, cast

import redis.asyncio as redis  # type: ignore[import-untyped]
from injector import inject, singleton
from pydantic import BaseModel, Field

from private_gpt.arq.settings import get_redis_settings
from private_gpt.components.engines.chat.async_chat_engine import (
    IterationCheckpointPayload,
)
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.settings.settings import Settings


class IterationContext(BaseModel):
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


@singleton
class IterationStateService:
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
        self._ttl = 3600

    def _ctx_key(self, correlation_id: str) -> str:
        return f"{self._prefix}:ctx:{correlation_id}"

    def _results_key(self, correlation_id: str) -> str:
        return f"{self._prefix}:results:{correlation_id}"

    def _resumed_key(self, correlation_id: str) -> str:
        return f"{self._prefix}:resumed:{correlation_id}"

    async def save(self, ctx: IterationContext) -> None:
        await self._redis.set(
            self._ctx_key(ctx.correlation_id), ctx.model_dump_json(), ex=self._ttl
        )
        await self._redis.delete(
            self._results_key(ctx.correlation_id), self._resumed_key(ctx.correlation_id)
        )

    async def load(self, correlation_id: str) -> IterationContext | None:
        data = await self._redis.get(self._ctx_key(correlation_id))
        if not data:
            return None
        return IterationContext.model_validate_json(data)

    async def record_result(
        self, correlation_id: str, tool_id: str, result: dict[str, Any]
    ) -> dict[str, ToolExecutionResponse] | None:
        ctx = await self.load(correlation_id)
        if ctx is None:
            return None

        results_key = self._results_key(correlation_id)
        await cast(Any, self._redis.hset(results_key, tool_id, json.dumps(result)))
        await cast(Any, self._redis.expire(results_key, self._ttl))
        pending_async_tools = ctx.checkpoint_payload.pending_async_tools
        if cast(int, await cast(Any, self._redis.hlen(results_key))) < len(
            pending_async_tools
        ):
            return None

        payload = cast(
            dict[str, str], await cast(Any, self._redis.hgetall(results_key))
        )
        return {
            key: ToolExecutionResponse.model_validate_json(value)
            for key, value in payload.items()
        }

    async def claim_resume(self, correlation_id: str) -> bool:
        return bool(
            await self._redis.set(
                self._resumed_key(correlation_id), "1", ex=self._ttl, nx=True
            )
        )

    async def cleanup(self, correlation_id: str) -> None:
        await self._redis.delete(
            self._ctx_key(correlation_id),
            self._results_key(correlation_id),
            self._resumed_key(correlation_id),
        )

    async def close(self) -> None:
        await self._redis.aclose()
