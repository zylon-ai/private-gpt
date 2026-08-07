"""Event models for the ZGPT file callback system."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

FileEventType = Literal["file.created", "file.updated", "file.deleted"]


class FileEvent(BaseModel):
    """A typed filesystem event emitted when a container-visible file changes."""

    type: FileEventType = Field(description="Event discriminator.")
    path: str = Field(description="Relative path within the scope that changed.")
    namespace: str = Field(description="Namespace that owns the file.")
    scope: str = Field(description="Scope identifier within the namespace.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="UTC timestamp of the event.",
    )
    correlation: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-supplied correlation properties passed through unchanged.",
    )


class AmqpCallbackTarget(BaseModel):
    exchange: str = Field(description="AMQP exchange name.")
    routing_key: str = Field(description="Routing key for event messages.")


class HttpCallbackTarget(BaseModel):
    url: str = Field(description="HTTP endpoint URL that receives POST events.")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers to include in the POST request.",
    )


class FileCallbackTarget(BaseModel):
    """Where to send file events for a given watch session."""

    amqp: AmqpCallbackTarget | None = Field(
        default=None,
        description="Publish events to an AMQP exchange.",
    )
    http: HttpCallbackTarget | None = Field(
        default=None,
        description="POST events to an HTTP endpoint.",
    )
    correlation: dict[str, Any] = Field(
        default_factory=dict,
        description="Properties merged into every emitted event.",
    )
