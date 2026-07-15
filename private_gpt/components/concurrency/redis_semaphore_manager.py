"""Async priority queue with distributed concurrency control using Redis."""
import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from redis.asyncio import Redis  # type: ignore[import-untyped]
from redis_semaphore_async import Semaphore

from private_gpt.components.concurrency.semaphore_manager import (
    QueueShutdownError,
    SemaphoreManager,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class RedisSemaphoreManager(SemaphoreManager):
    def __init__(
        self,
        settings: Settings | None = None,
        max_concurrency: int | None = None,
        queue_key: str | None = None,
    ) -> None:
        settings = settings or get_global_injector().get(Settings)
        self.redis_url: str = f"{settings.redis.url}/12"
        self.max_concurrency: int = max_concurrency or 2

        shared_name = queue_key or "default"
        self.semaphore_name: str = f"{shared_name}:semaphore"

        instance_id = str(uuid.uuid4())[:8]
        self.queue_key: str = f"queue:{shared_name}:{instance_id}"

        self.redis: Redis | None = None
        self._redis_lock = asyncio.Lock()
        self.semaphore: Semaphore | None = None
        self._tasks: dict[
            str,
            tuple[Callable[..., Awaitable[Any]], dict[str, Any], asyncio.Future[Any]],
        ] = {}
        self._worker: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._started = False

    async def __aenter__(self) -> "RedisSemaphoreManager":
        await self.start_processor()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _get_redis(self) -> Redis:
        if self.redis is None:
            async with self._redis_lock:
                if self.redis is None:
                    self.redis = Redis.from_url(
                        self.redis_url, encoding="utf-8", decode_responses=True
                    )
        return self.redis

    async def _get_semaphore(self) -> Semaphore:
        if self.semaphore is None:
            redis = await self._get_redis()
            self.semaphore = Semaphore(
                redis=redis,
                task_name=self.semaphore_name,
                value=self.max_concurrency,
                namespace="queue_semaphore",
            )
        return self.semaphore

    async def _dequeue_task(self) -> tuple[str, int] | None:
        redis = await self._get_redis()

        script = """
        local items = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
        if #items == 0 then
            return nil
        end
        redis.call('ZREM', KEYS[1], items[1])
        return {items[1], items[2]}
        """

        result = await redis.eval(script, 1, self.queue_key)  # type: ignore

        if not result:
            return None

        payload_str, priority = result
        payload = json.loads(payload_str)
        return payload["request_id"], int(float(priority))

    async def _execute_task(
        self,
        request_id: str,
        priority: int,
    ) -> None:
        task_info = self._tasks.pop(request_id, None)
        if not task_info:
            logger.error("Task %s not found in local registry", request_id)
            return

        func, kwargs, future = task_info

        try:
            logger.debug(
                "Executing task %s (priority=%d)",
                request_id,
                priority,
            )
            result = await func(**kwargs)

            if not future.done():
                future.set_result(result)

        except Exception as e:
            logger.error("Task %s failed: %s", request_id, e, exc_info=True)
            if not future.done():
                future.set_exception(e)

    async def _worker_loop(self) -> None:
        logger.debug("Worker started for queue: %s", self.queue_key)
        semaphore = await self._get_semaphore()

        while not self._shutdown.is_set():
            try:
                task_data = await self._dequeue_task()
                if task_data is None:
                    await asyncio.sleep(0.5)
                    continue

                request_id, priority = task_data
                logger.debug("Waiting for semaphore slot for task %s", request_id)

                acquired = False
                try:
                    acquired = await semaphore.acquire()
                    if not acquired:
                        logger.error(
                            "Failed to acquire semaphore for task %s, re-enqueueing",
                            request_id,
                        )
                        redis = await self._get_redis()
                        payload_str = json.dumps(
                            {"request_id": request_id, "priority": priority}
                        )
                        await redis.zadd(self.queue_key, {payload_str: priority})
                        continue

                    if self._shutdown.is_set():
                        logger.debug(
                            "Shutdown during acquire, re-enqueueing task %s", request_id
                        )
                        redis = await self._get_redis()
                        payload_str = json.dumps(
                            {"request_id": request_id, "priority": priority}
                        )
                        await redis.zadd(self.queue_key, {payload_str: priority})
                        break

                    await self._execute_task(request_id, priority)

                finally:
                    if acquired:
                        await semaphore.release()
                        logger.debug("Released semaphore for task %s", request_id)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("Worker error: %s", e, exc_info=True)
                await asyncio.sleep(1)

        logger.debug("Worker stopped for queue: %s", self.queue_key)

    async def start_processor(self) -> None:
        if self._started:
            return

        self._started = True
        self._shutdown.clear()

        self._worker = asyncio.create_task(self._worker_loop())
        logger.info(
            "Started worker for queue %s (semaphore max=%d)",
            self.queue_key,
            self.max_concurrency,
        )

    async def execute(
        self,
        task_func: Callable[..., Awaitable[Any]],
        priority: int = 0,
        **kwargs: Any,
    ) -> Any:
        if self._shutdown.is_set():
            raise QueueShutdownError("Queue is shutting down")

        redis = await self._get_redis()
        request_id = str(uuid.uuid4())
        future: asyncio.Future[Any] = asyncio.Future()
        payload_str = json.dumps({"request_id": request_id, "priority": priority})

        await self.start_processor()

        if self._shutdown.is_set():
            raise QueueShutdownError("Queue is shutting down")

        self._tasks[request_id] = (task_func, kwargs, future)

        await redis.zadd(self.queue_key, {payload_str: priority})
        logger.debug("Enqueued task %s (priority=%d)", request_id, priority)

        try:
            return await future

        except asyncio.CancelledError:
            self._tasks.pop(request_id, None)
            await redis.zrem(self.queue_key, payload_str)
            raise

    async def close(self) -> None:
        logger.info("Closing queue: %s", self.queue_key)

        self._shutdown.set()

        for _, (_, _, future) in list(self._tasks.items()):
            if not future.done():
                future.cancel()
        self._tasks.clear()

        if self._worker:
            self._worker.cancel()

            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._worker, timeout=30.0)

            self._worker = None

        if self.redis:
            await self.redis.aclose()
            self.redis = None

        logger.info("Queue closed: %s", self.queue_key)
