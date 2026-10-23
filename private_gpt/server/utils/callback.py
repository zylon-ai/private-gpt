from typing import Any

from pydantic import BaseModel, Field


class AMQP(BaseModel):
    """AMQP configuration for message queue notifications."""

    exchange: str = Field(
        examples=["ingest"],
        description="AMQP exchange name where task notifications will be published",
    )
    routing_key_done: str | None = Field(
        default=None,
        examples=["ingest.done"],
        description="Routing key used when publishing task completion notifications",
    )
    routing_key_progress: str | None = Field(
        default=None,
        examples=["ingest.progress"],
        description="Routing key used when publishing task progress update notifications",
    )
    routing_key_error: str | None = Field(
        default=None,
        examples=["ingest.error"],
        description="Routing key used when publishing task error notifications",
    )


class Callback(BaseModel):
    """Callback configuration for asynchronous task notifications."""

    amqp: AMQP = Field(
        examples=[
            {
                "exchange": "ingest",
                "routing_key_done": "ingest.done",
                "routing_key_error": "ingest.error",
            }
        ],
        description="AMQP (Advanced Message Queuing Protocol) configuration for task status notifications",
    )
    properties: dict[str, Any] | None = Field(
        default=None,
        examples=[{"key": "value"}],
        description="Additional properties and metadata to include with callback notifications",
    )


class BaseCallbackInput(BaseModel):
    """Base model for requests that support asynchronous callback notifications."""

    callback: Callback | None = Field(
        default=None,
        examples=[
            {
                "amqp": {
                    "exchange": "ingest",
                    "routing_key_done": "ingest.done",
                    "routing_key_error": "ingest.error",
                },
                "properties": {"key": "value"},
            }
        ],
        description="Optional callback configuration for receiving asynchronous task notifications",
    )


class AsyncResponse(BaseModel):
    """Response containing all the info to be sent to the callback."""

    data: Any = Field(
        description="The response data payload from the asynchronous operation"
    )
    type: str = Field(
        examples=["summary_index_task"],
        description="Type identifier for the asynchronous operation that completed",
    )
    error: Any | None = Field(
        default=None,
        description="Error information if the asynchronous operation failed, null if successful",
    )
    callback_properties: dict[str, Any] | None = Field(
        examples=[{"key": "value"}],
        default=None,
        description="Additional properties and metadata included from the original callback configuration",
    )
