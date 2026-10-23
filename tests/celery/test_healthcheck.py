import pytest

from private_gpt.celery import healthcheck


@pytest.mark.parametrize("mode", ["", "mixed", "worker", "unknown"])
async def test_healthcheck_rejects_unregistered_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("PGPT_WORKER_MODE", mode)

    status = await healthcheck.health_check()

    assert status["status"] == "unhealthy"
    assert status["services"] == {}


async def test_healthcheck_checks_only_celery_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_WORKER_MODE", "celery")
    monkeypatch.setattr(healthcheck, "check_worker", lambda: _async_result(True))

    status = await healthcheck.health_check()

    assert status["status"] == "healthy"
    assert status["services"] == {"worker": "healthy"}


async def _async_result(value: bool) -> bool:
    return value
