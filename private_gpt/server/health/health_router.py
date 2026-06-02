from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

# Not authentication or authorization required to get the health status.
health_router = APIRouter(
    include_in_schema=False,
)


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(default="ok")


@health_router.get("/health", tags=["Health"])
async def health() -> HealthResponse:
    """Return ok if the system is up."""
    return HealthResponse(status="ok")
