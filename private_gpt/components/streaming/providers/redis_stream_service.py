import asyncio
import contextlib
import json
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, cast

import redis.asyncio as redis  # type: ignore[import-untyped]
from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import (
    StreamMetadata,
    StreamStatus,
)
from private_gpt.components.streaming.providers.stream_service import (
    Event,
    StreamService,
)


class RedisStreamConfig(BaseModel):
    redis_url: str
    stream_prefix: str
    status_prefix: str
    expiry_seconds: int
    max_stream_length: int
    minimum_connections: int | None


class RedisStreamService(StreamService):
    """Minimal Redis-based streaming service for raw event data."""

    _config: RedisStreamConfig
    _client: redis.Redis

    def __init__(self, config: RedisStreamConfig, redis_client: redis.Redis) -> None:
        self._config = config
        self._client = redis_client

        if config.minimum_connections is not None:
            task = asyncio.create_task(
                self._pre_create_connections(redis_client, config.minimum_connections)
            )
            task.add_done_callback(self._consume_background_result)

    @staticmethod
    def _consume_background_result(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

    async def _pre_create_connections(self, client: redis.Redis, n: int) -> None:
        client.auto_close_connection_pool = False
        client.single_connection_client = False

        async def create_connection(connection_pool: redis.ConnectionPool) -> None:
            connection = connection_pool.make_connection()
            await connection_pool.ensure_connection(connection)
            connection_pool._available_connections.append(connection)

        connection_pool = cast(redis.ConnectionPool, client.connection_pool)
        tasks = [create_connection(connection_pool) for _ in range(n)]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _get_stream_key(self, correlation_id: str) -> str:
        """Generate stream key for correlation ID."""
        return f"{self._config.stream_prefix}:{correlation_id}"

    def _get_status_key(self, correlation_id: str) -> str:
        """Generate status key for correlation ID."""
        return f"{self._config.status_prefix}:{correlation_id}"

    def _get_cancel_key(self, correlation_id: str) -> str:
        """Generate cancel-flag key for correlation ID."""
        return f"{self._config.status_prefix}:cancel:{correlation_id}"

    async def create_stream(
        self,
        stream_type: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new stream and return correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        now = datetime.now(UTC)
        stream_metadata = StreamMetadata(
            correlation_id=correlation_id,
            status=StreamStatus.PENDING,
            created_at=now,
            updated_at=now,
            stream_type=stream_type,
            metadata=metadata or {},
        )

        mapping = await asyncio.to_thread(stream_metadata.model_dump_json_fields)
        status_key = self._get_status_key(correlation_id)
        created = await self._client.eval(
            """
            if redis.call('exists', KEYS[1]) == 1 then
                return 0
            end
            local fields = cjson.decode(ARGV[1])
            for key, value in pairs(fields) do
                redis.call('hset', KEYS[1], key, value)
            end
            redis.call('expire', KEYS[1], ARGV[2])
            return 1
            """,
            1,
            status_key,
            json.dumps(mapping),
            self._config.expiry_seconds,
        )
        if not created:
            raise ValueError(
                f"Stream with correlation_id {correlation_id} already exists"
            )

        return correlation_id

    async def update_stream_status(
        self,
        correlation_id: str,
        status: StreamStatus,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update stream status and metadata."""
        status_key = self._get_status_key(correlation_id)

        now = datetime.now(UTC)
        updates = {
            "status": status.value,
            "updated_at": now.isoformat(),
        }

        if error_message:
            updates["error_message"] = error_message

        if status in [
            StreamStatus.COMPLETED,
            StreamStatus.CANCELLED,
            StreamStatus.ERROR,
        ]:
            updates["completed_at"] = now.isoformat()

        if metadata:
            existing_data = await self._client.hget(status_key, "metadata")  # type: ignore

            def process_metadata() -> str:
                existing_metadata = json.loads(existing_data) if existing_data else {}
                existing_metadata.update(metadata)
                return json.dumps(existing_metadata)

            updates["metadata"] = await asyncio.to_thread(process_metadata)

        await self._client.hset(status_key, mapping=updates)  # type: ignore
        await self._client.expire(status_key, self._config.expiry_seconds)

    async def get_stream_metadata(self, correlation_id: str) -> StreamMetadata | None:
        """Get stream metadata by correlation ID."""
        status_key = self._get_status_key(correlation_id)
        data = cast(dict[str, str], await self._client.hgetall(status_key))  # type: ignore
        if not data:
            return None

        def deserialize_metadata() -> StreamMetadata:
            result = StreamMetadata(
                correlation_id=data["correlation_id"],
                status=StreamStatus.from_string(str(data["status"]).lower())
                if isinstance(data["status"], str)
                else data["status"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                completed_at=datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None,
                error_message=data.get("error_message"),
                stream_type=data["stream_type"],
                metadata=json.loads(data.get("metadata", "{}")),
            )
            return result

        result = deserialize_metadata()
        return result

    async def push_event(
        self,
        correlation_id: str,
        event_data: str,
    ) -> str:
        """Push raw event data to the stream."""
        stream_key = self._get_stream_key(correlation_id)
        event_fields: dict[str, str] = {"data": event_data}

        async with self._client.pipeline(transaction=False) as pipe:
            await pipe.xadd(
                stream_key,
                event_fields,  # type: ignore
                maxlen=self._config.max_stream_length,
            )
            await pipe.expire(stream_key, self._config.expiry_seconds)
            results = await pipe.execute()

        return str(results[0])

    async def push_event_batch(self, events: list[Event]) -> dict[str, str]:
        """Push multiple events efficiently using pipeline."""
        if not events:
            return {}

        grouped: dict[str, list[str]] = defaultdict(list)
        for event in events:
            grouped[event.correlation_id].append(event.event_data)

        async with self._client.pipeline(transaction=False) as pipe:
            result_indices: dict[str, int] = {}
            idx = 0

            for correlation_id, event_list in grouped.items():
                stream_key = self._get_stream_key(correlation_id)
                for event_data in event_list:
                    await pipe.xadd(
                        stream_key,
                        {"data": event_data},
                        maxlen=self._config.max_stream_length,
                    )
                    idx += 1
                result_indices[correlation_id] = idx - 1
                await pipe.expire(stream_key, self._config.expiry_seconds)
                idx += 1

            results = await pipe.execute()

        return {
            correlation_id: str(results[result_idx])
            for correlation_id, result_idx in result_indices.items()
        }

    async def read_events(
        self,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[str], str]:
        """Read raw event data from stream and return next last_id."""
        stream_key = self._get_stream_key(correlation_id)
        events = await self._client.xread(
            {stream_key: last_id}, count=count, block=block_ms
        )

        event_data = []
        next_last_id = last_id

        for _, msgs in events:
            for msg_id, fields in msgs:
                event_data.append(fields.get("data", ""))
                next_last_id = msg_id  # Use actual Redis message ID

        return event_data, next_last_id

    async def stream_exists(self, correlation_id: str) -> bool:
        """Check if stream exists."""
        status_key = self._get_status_key(correlation_id)
        result = await self._client.exists(status_key)
        return bool(result)

    async def delete_stream(self, correlation_id: str) -> None:
        """Delete stream and its metadata."""
        stream_key = self._get_stream_key(correlation_id)
        status_key = self._get_status_key(correlation_id)

        await self._client.delete(stream_key, status_key)

    async def list_streams(
        self,
        stream_type: str | None = None,
        status: StreamStatus | None = None,
        limit: int | None = None,
    ) -> list[StreamMetadata]:
        """List streams with optional filtering."""
        pattern = f"{self._config.status_prefix}:*"
        keys = await self._client.keys(pattern)

        async def deserialize_stream(key: str) -> StreamMetadata | None:
            try:
                data = cast(dict[str, str], await self._client.hgetall(key))  # type: ignore
                if not data:
                    return None

                def build_metadata() -> StreamMetadata:
                    return StreamMetadata(
                        correlation_id=data["correlation_id"],
                        status=StreamStatus.from_string(data["status"]),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        completed_at=datetime.fromisoformat(data["completed_at"])
                        if data.get("completed_at")
                        else None,
                        error_message=data.get("error_message"),
                        stream_type=data["stream_type"],
                        metadata=json.loads(data.get("metadata", "{}")),
                    )

                return await asyncio.to_thread(build_metadata)
            except Exception:
                return None

        streams = await asyncio.gather(
            *[deserialize_stream(key) for key in keys[:limit]], return_exceptions=True
        )

        valid_streams = [
            s
            for s in streams
            if isinstance(s, StreamMetadata)
            and (not stream_type or s.stream_type == stream_type)
            and (not status or s.status == status)
        ]

        return sorted(valid_streams, key=lambda x: x.created_at, reverse=True)

    async def clean_up_stream(self, correlation_id: str) -> None:
        """Clean up a stream by deleting it and its events."""
        await self.delete_stream(correlation_id)
        await self.clear_cancel_flag(correlation_id)

    async def set_cancel_flag(self, correlation_id: str) -> None:
        """Set a cancellation flag in Redis for the chat worker to observe."""
        cancel_key = self._get_cancel_key(correlation_id)
        await self._client.set(cancel_key, "1", ex=self._config.expiry_seconds)

    async def is_cancelled(self, correlation_id: str) -> bool:
        """Check whether the cancellation flag has been set."""
        cancel_key = self._get_cancel_key(correlation_id)
        return bool(await self._client.exists(cancel_key))

    async def clear_cancel_flag(self, correlation_id: str) -> None:
        """Remove the cancellation flag."""
        cancel_key = self._get_cancel_key(correlation_id)
        await self._client.delete(cancel_key)

    async def close(self) -> None:
        """Close the redis stream service."""
        pass  # Do nothing, as the client is managed by the factory


class LazyRedisClientFactory:
    """Lazy initialization of Redis client."""

    _instance: redis.Redis | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, redis_config: RedisStreamConfig) -> redis.Redis:
        """Get a Redis client instance, initializing it if necessary."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = redis.Redis.from_url(
                        redis_config.redis_url,
                        decode_responses=True,
                        socket_timeout=5.0,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=30,
                        retry_on_timeout=True,
                    )
        return cls._instance

    @classmethod
    def close_instance(cls) -> None:
        """Close the Redis client instance if it exists."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
