from typing import Any

from fastapi import APIRouter

webhook_router = APIRouter()


@webhook_router.post("/webhook")
def register_webhook() -> Any:
    return {"message": "TODO: Not implemented yet"}
