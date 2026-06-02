import json
from typing import cast

from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.events.models import (
    Event,
    FatalError,
    PingEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)


class StreamingEventHandler:
    """Event handler for the Event union type."""

    def __init__(self) -> None:
        # Map discriminator values to their corresponding classes
        self.event_classes: dict[str, type[BaseModel]] = {
            "content_block_start": RawContentBlockStartEvent,
            "content_block_delta": RawContentBlockDeltaEvent,
            "content_block_stop": RawContentBlockStopEvent,
            "message_start": RawMessageStartEvent,
            "message_delta": RawMessageDeltaEvent,
            "message_stop": RawMessageStopEvent,
            "ping": PingEvent,
            "error": FatalError,
        }

    def serialize(self, event: BaseModel) -> str:
        """Serialize event to minimal JSON string."""
        return event.model_dump_json()

    def deserialize(self, data: str) -> Event:
        """Deserialize JSON string back to correct Event type."""
        try:
            event_dict = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {data}") from e

        event_type = event_dict.get("type")
        if not event_type:
            raise ValueError("Event data missing 'type' field")

        event_class = self.event_classes.get(event_type)
        if not event_class:
            raise ValueError(f"Unknown event type: {event_type}")

        try:
            return cast(Event, event_class.model_validate(event_dict))
        except Exception as e:
            raise ValueError(
                f"Failed to deserialize event of type {event_type}: {e}"
            ) from e

    async def get_current_status(self, event: BaseModel) -> StreamStatus | None:
        """Check if the stream is currently being processed."""
        if isinstance(event, RawMessageStartEvent):
            return StreamStatus.PENDING
        elif isinstance(
            event,
            RawContentBlockStartEvent
            | RawContentBlockDeltaEvent
            | RawContentBlockStopEvent,
        ):
            return StreamStatus.PROCESSING
        elif isinstance(event, RawMessageStopEvent):
            return StreamStatus.COMPLETED
        elif isinstance(event, FatalError):
            return StreamStatus.FAILED

        return None

    def error_event(self, correlation_id: str, error: Exception) -> FatalError:
        """Convert an Exception to a serializable error event."""
        return FatalError.from_exception(error, request_id=correlation_id)
