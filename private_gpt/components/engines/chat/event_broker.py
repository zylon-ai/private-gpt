from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, cast

import redis.asyncio as redis  # type: ignore[import-untyped]
from injector import Injector, inject, singleton

from private_gpt.arq.settings import get_redis_settings
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from private_gpt.events.models import Event


class EngineEventBroker(ABC):
    """Transport used between chat execution and router-side stream processing."""

    @abstractmethod
    async def publish(self, execution_id: str, event: Event) -> None:
        ...

    @abstractmethod
    async def finish(self, execution_id: str) -> None:
        ...

    @abstractmethod
    def listen(self, execution_id: str) -> AsyncGenerator[Event, None]:
        ...

    @abstractmethod
    async def cleanup(self, execution_id: str) -> None:
        ...


@singleton
class InMemoryEngineEventBroker(EngineEventBroker):
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Event | None]] = defaultdict(
            asyncio.Queue
        )

    async def publish(self, execution_id: str, event: Event) -> None:
        await self._queues[execution_id].put(event)

    async def finish(self, execution_id: str) -> None:
        await self._queues[execution_id].put(None)

    async def listen(self, execution_id: str) -> AsyncGenerator[Event, None]:
        queue = self._queues[execution_id]
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._queues.pop(execution_id, None)

    async def cleanup(self, execution_id: str) -> None:
        self._queues.pop(execution_id, None)


@singleton
class RedisEngineEventBroker(EngineEventBroker):
    _FINISHED = "__private_gpt_engine_finished__"

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
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self._handler = StreamingEventHandler()
        self._prefix = "private_gpt:engine:events"
        self._ttl = settings.stream.stream_expiration

    def _key(self, execution_id: str) -> str:
        return f"{self._prefix}:{execution_id}"

    async def publish(self, execution_id: str, event: Event) -> None:
        key = self._key(execution_id)
        await cast(
            Awaitable[int], self._redis.rpush(key, self._handler.serialize(event))
        )
        await cast(Awaitable[int], self._redis.expire(key, self._ttl))

    async def finish(self, execution_id: str) -> None:
        key = self._key(execution_id)
        await cast(Awaitable[int], self._redis.rpush(key, self._FINISHED))
        await cast(Awaitable[int], self._redis.expire(key, self._ttl))

    async def listen(self, execution_id: str) -> AsyncGenerator[Event, None]:
        key = self._key(execution_id)
        try:
            while True:
                item = cast(
                    tuple[str, str] | None,
                    await cast(Awaitable[Any], self._redis.blpop([key], timeout=1)),
                )
                if item is None:
                    continue
                payload = item[1]
                if payload == self._FINISHED:
                    return
                yield self._handler.deserialize(payload)
        finally:
            await self.cleanup(execution_id)

    async def cleanup(self, execution_id: str) -> None:
        await self._redis.delete(self._key(execution_id))

    async def close(self) -> None:
        await self._redis.aclose()


@singleton
class EngineEventBrokerFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector

    def get(self) -> EngineEventBroker:
        if self._settings.scheduler.chat.mode == "arq":
            return self._injector.get(RedisEngineEventBroker)
        return self._injector.get(InMemoryEngineEventBroker)
