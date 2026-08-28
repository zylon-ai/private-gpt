from private_gpt.arq.healthcheck import health


async def test_healthcheck_does_not_require_redis() -> None:
    assert await health() == {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
    }
