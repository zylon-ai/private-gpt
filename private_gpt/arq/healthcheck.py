from typing import Any

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
    }
