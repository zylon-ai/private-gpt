from fastapi import APIRouter
from pydantic import BaseModel, Field

health_router = APIRouter()


class HealthResponse(BaseModel):
    status: str = Field(enum=["ok"])


@health_router.get("/health", tags=["Health"])
def health() -> HealthResponse:
    """Return ok if the system is up."""
    return HealthResponse(status="ok")
