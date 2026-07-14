from typing import Any

from arq import create_pool
from fastapi import FastAPI, HTTPException

from private_gpt.arq.settings import get_health_check_key, get_redis_settings
from private_gpt.settings.settings import settings

app = FastAPI()


@app.get("/health")
async def health() -> dict[str, Any]:
    current_settings = settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        healthy = bool(await redis.exists(get_health_check_key(current_settings)))
    finally:
        await redis.aclose()

    status = {
        "status": "healthy" if healthy else "unhealthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy" if healthy else "unhealthy"},
    }
    if not healthy:
        raise HTTPException(status_code=503, detail=status)
    return status
