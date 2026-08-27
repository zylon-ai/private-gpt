from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arq.connections import create_pool

from private_gpt.arq.settings import (
    get_control_redis_settings,
    get_redis_settings,
)

if TYPE_CHECKING:
    from arq.connections import ArqRedis

    from private_gpt.settings.settings import Settings


ROUTE_KEY_PREFIX = "private_gpt:arq:publisher-route"
ROUTE_TTL_SECONDS = 86400


@dataclass(frozen=True)
class PublisherRoute:
    queue_name: str
    database: int
    updated_at_ms: int


def route_key(worker_type: str) -> str:
    return f"{ROUTE_KEY_PREFIX}:{worker_type}"


def expected_queue_for_worker(settings: Settings, worker_type: str) -> str | None:
    scheduler = getattr(settings.scheduler, worker_type, None)
    queue = getattr(scheduler, "celery_queue", None)
    if not isinstance(queue, str) or not queue.strip():
        return None
    from private_gpt.arq.settings import get_queue_name

    return get_queue_name(queue.strip())


def encode_route(route: PublisherRoute) -> str:
    return json.dumps(
        {
            "queue_name": route.queue_name,
            "database": route.database,
            "updated_at_ms": route.updated_at_ms,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_route(value: bytes | str) -> PublisherRoute:
    payload: dict[str, Any] = json.loads(
        value.decode() if isinstance(value, bytes) else value
    )
    return PublisherRoute(
        queue_name=str(payload["queue_name"]),
        database=int(payload["database"]),
        updated_at_ms=int(payload["updated_at_ms"]),
    )


async def publish_route(
    settings: Settings,
    *,
    worker_type: str,
    queue_name: str,
) -> None:
    redis: ArqRedis | None = None
    try:
        redis = await create_pool(get_control_redis_settings(settings))
        route = PublisherRoute(
            queue_name=queue_name,
            database=get_redis_settings(settings).database,
            updated_at_ms=int(time.time() * 1000),
        )
        await redis.setex(
            route_key(worker_type),
            ROUTE_TTL_SECONDS,
            encode_route(route),
        )
    finally:
        if redis is not None:
            await redis.aclose()
