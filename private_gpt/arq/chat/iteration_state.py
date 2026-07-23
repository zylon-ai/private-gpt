"""Redis-backed state for resumable ARQ chat execution."""

from __future__ import annotations

from typing import Any, cast

import redis.asyncio as redis  # type: ignore[import-untyped]
from injector import inject, singleton

from private_gpt.arq.settings import get_redis_settings
from private_gpt.components.engines.chat.checkpoint_store import (
    ChatCheckpoint,
    ChatCheckpointStore,
    normalize_tool_result,
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

    def _actions_key(self, execution_id: str) -> str:
        return f"{self._prefix}:actions:{execution_id}"

    def _terminal_key(self, execution_id: str) -> str:
        return f"{self._prefix}:terminal:{execution_id}"

    async def save(self, checkpoint: ChatCheckpoint) -> bool:
        saved = await cast(Any, self._redis.eval)(
            """
            if redis.call('exists', KEYS[1]) == 1 then
                return 0
            end
            redis.call('set', KEYS[2], ARGV[1], 'EX', ARGV[2])
            redis.call('del', KEYS[3])
            return 1
            """,
            3,
            self._terminal_key(checkpoint.correlation_id),
            self._ctx_key(checkpoint.correlation_id),
            self._resumed_key(checkpoint.correlation_id),
            checkpoint.model_dump_json(),
            self._ttl,
        )
        return bool(saved)

    async def load(self, execution_id: str) -> ChatCheckpoint | None:
        data = await self._redis.get(self._ctx_key(execution_id))
        return ChatCheckpoint.model_validate_json(data) if data else None

    async def record_result(
        self,
        execution_id: str,
        tool_id: str,
        result: dict[str, Any],
        *,
        allow_claimed: bool = False,
    ) -> dict[str, ToolExecutionResponse] | None:
        response = normalize_tool_result(tool_id, result)
        results_key = self._results_key(execution_id)
        recorded = await cast(Any, self._redis.eval)(
            """
            if redis.call('exists', KEYS[1]) == 1 then
                return 0
            end
            if ARGV[4] == '0' and redis.call('exists', KEYS[2]) == 1 then
                return 0
            end
            if redis.call('hsetnx', KEYS[3], ARGV[1], ARGV[2]) == 0 then
                return 0
            end
            redis.call('expire', KEYS[3], ARGV[3])
            return 1
            """,
            3,
            self._terminal_key(execution_id),
            self._resumed_key(execution_id),
            results_key,
            tool_id,
            response.model_dump_json(),
            self._ttl,
            "1" if allow_claimed else "0",
        )
        if not recorded:
            return None
        checkpoint = await self.load(execution_id)
        if checkpoint is None:
            return None
        expected = set(checkpoint.checkpoint_payload.pending_async_tools)
        if tool_id not in expected:
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

    async def claim_action(self, execution_id: str, action_id: str) -> bool:
        key = self._actions_key(execution_id)
        claimed = await cast(Any, self._redis.hsetnx(key, action_id, "1"))
        await cast(Any, self._redis.expire(key, self._ttl))
        return bool(claimed)

    async def mark_terminal(self, execution_id: str, status: str) -> bool:
        return bool(
            await self._redis.set(
                self._terminal_key(execution_id),
                status,
                ex=self._ttl,
                nx=True,
            )
        )

    async def release_resume(self, execution_id: str) -> None:
        await self._redis.delete(self._resumed_key(execution_id))

    async def cleanup(self, execution_id: str) -> None:
        await self._redis.delete(
            self._ctx_key(execution_id),
            self._results_key(execution_id),
            self._resumed_key(execution_id),
        )

    async def close(self) -> None:
        await self._redis.aclose()
