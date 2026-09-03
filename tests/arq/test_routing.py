from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    import pytest

from private_gpt.arq import routing
from private_gpt.arq.routing import (
    PublisherRoute,
    decode_route,
    encode_route,
    expected_queue_for_worker,
    publish_route,
)


def test_route_round_trip() -> None:
    route = PublisherRoute(
        queue_name="private_gpt:arq:queue:chat",
        database=8,
        updated_at_ms=123,
    )

    assert decode_route(encode_route(route)) == route


def test_expected_queue_for_worker() -> None:
    settings = MagicMock()
    settings.scheduler.chat.celery_queue = "chat"

    assert expected_queue_for_worker(settings, "chat") == ("private_gpt:arq:queue:chat")


async def test_publish_route_uses_control_db_and_actual_arq_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MagicMock()
    control = MagicMock(database=0)
    arq_settings = MagicMock(database=8)
    redis = MagicMock()
    redis.setex = AsyncMock()
    redis.aclose = AsyncMock()
    monkeypatch.setattr(routing, "get_control_redis_settings", lambda _: control)
    monkeypatch.setattr(routing, "get_redis_settings", lambda _: arq_settings)
    create_pool = AsyncMock(return_value=redis)
    monkeypatch.setattr(routing, "create_pool", create_pool)

    await publish_route(
        settings,
        worker_type="chat",
        queue_name="private_gpt:arq:queue:chat",
    )

    create_pool.assert_awaited_once_with(control)
    redis.setex.assert_awaited_once()
    key, ttl, raw = redis.setex.await_args.args
    assert key == "private_gpt:arq:publisher-route:chat"
    assert ttl == routing.ROUTE_TTL_SECONDS
    assert decode_route(raw).database == 8
    await redis.aclose()
