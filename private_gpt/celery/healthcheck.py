import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from private_gpt.celery.bootsteps import HEARTBEAT_FILE, READINESS_FILE

HEALTHCHECK_TIMEOUT = 10


async def check_flower() -> bool:
    try:
        flower_port = os.getenv("PGPT_FLOWER_PORT", "5555")
        flower_prefix = os.getenv("PGPT_FLOWER_URL_PREFIX", "")
        url = f"http://localhost:{flower_port}{flower_prefix}/healthcheck"
        async with httpx.AsyncClient(timeout=HEALTHCHECK_TIMEOUT) as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception as e:
        logging.critical(f"Error checking Flower health: {e}")
        return False


async def check_worker() -> bool:
    try:
        if not READINESS_FILE.is_file():
            return False

        if not HEARTBEAT_FILE.is_file():
            return False

        stats = HEARTBEAT_FILE.stat()
        heartbeat_timestamp = stats.st_mtime
        current_timestamp = time.time()
        time_diff = current_timestamp - heartbeat_timestamp

        # Consider unhealthy if heartbeat is older than 60 seconds
        return time_diff <= 60
    except Exception as e:
        logging.critical(f"Error checking worker health: {e}")
        return False


async def health_check() -> dict[str, Any]:
    mode = os.getenv("PGPT_WORKER_MODE", "").strip().lower()
    status: dict[str, Any] = {"status": "healthy", "mode": mode, "services": {}}

    if mode == "flower":
        flower_health = await check_flower()
        status["services"]["flower"] = "healthy" if flower_health else "unhealthy"
        if not flower_health:
            status["status"] = "unhealthy"

    elif mode == "celery":
        worker_health = await check_worker()
        status["services"]["worker"] = "healthy" if worker_health else "unhealthy"
        if not worker_health:
            status["status"] = "unhealthy"

    else:
        status["status"] = "unhealthy"
        status["error"] = f"Unsupported worker mode: {mode or '<empty>'}"

    return status


app = FastAPI()


@app.get("/health")
async def health() -> dict[str, Any]:
    status = await health_check()
    if status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=status)
    return status
