from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from private_gpt.arq.healthcheck import check_health, health


@pytest.fixture(autouse=True)
def worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPT_ARQ_QUEUE", "chat")
    monkeypatch.setattr(
        "private_gpt.arq.healthcheck.load_settings",
        lambda: MagicMock(),
    )


def _patch_redis(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data: bytes | None = b"1",
    error: Exception | None = None,
) -> AsyncMock:
    redis = AsyncMock()
    if error is not None:
        redis.get = AsyncMock(side_effect=error)
    else:
        redis.get = AsyncMock(return_value=data)
    redis.aclose = AsyncMock()
    monkeypatch.setattr(
        "private_gpt.arq.healthcheck.create_pool",
        AsyncMock(return_value=redis),
    )
    return redis


async def test_healthcheck_is_healthy_when_arq_sentinel_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_redis(monkeypatch, data=b"1")

    status = await check_health()

    assert status == {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
    }


async def test_healthcheck_is_healthy_regardless_of_queued_or_ongoing_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_redis(
        monkeypatch,
        data=b"Mar-01 17:41:22 j_complete=0 j_failed=0 j_retried=0 j_ongoing=0 queued=99",
    )

    status = await check_health()

    assert status["status"] == "healthy"


async def test_healthcheck_is_unhealthy_when_arq_sentinel_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_redis(monkeypatch, data=None)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "arq_health_missing"


async def test_healthcheck_is_unhealthy_when_redis_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_redis(monkeypatch, error=ConnectionError("redis down"))

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "redis_unreachable"


async def test_healthcheck_is_unhealthy_when_queue_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPT_ARQ_QUEUE", raising=False)

    status = await check_health()

    assert status["status"] == "unhealthy"
    assert status["reason"] == "queue_not_configured"


async def test_healthcheck_uses_per_pod_arq_health_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOSTNAME", "chat-worker-pod")
    redis = _patch_redis(monkeypatch, data=b"1")
    redis_settings = object()
    monkeypatch.setattr(
        "private_gpt.arq.healthcheck.get_healthcheck_redis_settings",
        lambda _settings: redis_settings,
    )
    create_pool = AsyncMock(return_value=redis)
    monkeypatch.setattr("private_gpt.arq.healthcheck.create_pool", create_pool)

    await check_health()

    create_pool.assert_awaited_once_with(redis_settings)
    redis.get.assert_awaited_once_with(
        "private_gpt:arq:queue:chat:health-check:chat-worker-pod"
    )
    redis.aclose.assert_awaited_once()


async def test_http_health_returns_503_for_unhealthy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "private_gpt.arq.healthcheck.check_health",
        AsyncMock(return_value={"status": "unhealthy", "reason": "test"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await health()

    assert exc_info.value.status_code == 503
