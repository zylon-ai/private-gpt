from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.arq import enqueue


@pytest.fixture(autouse=True)
def _reset_shared_state() -> None:
    # enqueue keeps one pool per loop and throttles route publishes.
    enqueue._pools.clear()
    enqueue._route_published_at.clear()


async def test_enqueue_reuses_pool_and_throttles_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(enqueue, "_settings", MagicMock())
    redis = MagicMock()
    redis.enqueue_job = AsyncMock(return_value=object())
    create_pool = AsyncMock(return_value=redis)
    monkeypatch.setattr(enqueue, "create_pool", create_pool)
    publish_route = AsyncMock()
    monkeypatch.setattr(enqueue, "publish_route", publish_route)

    for _ in range(3):
        await enqueue.enqueue_job(
            task_name="private_gpt.chat.start",
            queue_name="private_gpt:arq:queue:chat",
            correlation_id="correlation-id",
            worker_type="chat",
        )

    assert create_pool.await_count == 1
    assert publish_route.await_count == 1
    assert redis.enqueue_job.await_count == 3


async def test_failed_route_publish_is_retried_on_next_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(enqueue, "_settings", MagicMock())
    redis = MagicMock()
    redis.enqueue_job = AsyncMock(return_value=object())
    monkeypatch.setattr(enqueue, "create_pool", AsyncMock(return_value=redis))
    publish_route = AsyncMock(side_effect=[RuntimeError("down"), None, None])
    monkeypatch.setattr(enqueue, "publish_route", publish_route)

    for _ in range(3):
        await enqueue.enqueue_job(
            task_name="private_gpt.chat.start",
            queue_name="private_gpt:arq:queue:chat",
            correlation_id="correlation-id",
            worker_type="chat",
        )

    assert publish_route.await_count == 2
    assert redis.enqueue_job.await_count == 3


def test_pools_of_closed_loops_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(enqueue, "_settings", MagicMock())
    pools = [MagicMock(), MagicMock()]
    monkeypatch.setattr(enqueue, "create_pool", AsyncMock(side_effect=pools))

    first_loop = asyncio.new_event_loop()
    assert first_loop.run_until_complete(enqueue._get_pool()) is pools[0]
    first_loop.close()

    second_loop = asyncio.new_event_loop()
    try:
        assert second_loop.run_until_complete(enqueue._get_pool()) is pools[1]
    finally:
        second_loop.close()
    assert first_loop not in enqueue._pools


async def test_enqueue_publishes_route_before_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_settings = MagicMock()
    monkeypatch.setattr(enqueue, "_settings", lambda: current_settings)
    redis = MagicMock()
    redis.enqueue_job = AsyncMock(return_value=object())
    redis.aclose = AsyncMock()
    monkeypatch.setattr(enqueue, "create_pool", AsyncMock(return_value=redis))
    publish_route = AsyncMock()
    monkeypatch.setattr(enqueue, "publish_route", publish_route)

    accepted = await enqueue.enqueue_job(
        task_name="private_gpt.chat.start",
        queue_name="private_gpt:arq:queue:chat",
        args=("payload",),
        correlation_id="correlation-id",
        worker_type="chat",
        job_id="job-id",
    )

    assert accepted is True
    publish_route.assert_awaited_once_with(
        current_settings,
        worker_type="chat",
        queue_name="private_gpt:arq:queue:chat",
    )
    redis.enqueue_job.assert_awaited_once_with(
        "private_gpt.chat.start",
        "payload",
        _queue_name="private_gpt:arq:queue:chat",
        _job_id="job-id",
    )


async def test_route_publish_failure_does_not_drop_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(enqueue, "_settings", MagicMock())
    redis = MagicMock()
    redis.enqueue_job = AsyncMock(return_value=object())
    redis.aclose = AsyncMock()
    monkeypatch.setattr(enqueue, "create_pool", AsyncMock(return_value=redis))
    monkeypatch.setattr(
        enqueue,
        "publish_route",
        AsyncMock(side_effect=RuntimeError("route registry unavailable")),
    )

    accepted = await enqueue.enqueue_job(
        task_name="private_gpt.chat.start",
        queue_name="private_gpt:arq:queue:chat",
        correlation_id="correlation-id",
        worker_type="chat",
    )

    assert accepted is True
    redis.enqueue_job.assert_awaited_once()
