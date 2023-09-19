"""Types for the API."""
from pydantic import BaseModel


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class HelloWorldRouteInput(BaseModel):
    """Model for the hello world route input."""

    name: str


class HelloWorldRouteOutput(BaseModel):
    """Model for the hello world route output."""

    message: str
