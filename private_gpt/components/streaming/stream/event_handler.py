from typing import Protocol

from pydantic import BaseModel

from private_gpt.components.streaming.providers.models import StreamStatus


class EventHandler(Protocol):
    """Protocol for handling event serialization/deserialization."""

    def serialize(self, event: BaseModel) -> str:
        """Serialize event to string for storage."""
        ...

    def deserialize(self, data: str) -> BaseModel:
        """Deserialize string data back to event."""
        ...

    async def get_current_status(self, event: BaseModel) -> StreamStatus | None:
        """Check if the stream is currently being processed."""
        ...

    def error_event(self, correlation_id: str, error: Exception) -> BaseModel:
        """Convert an Exception to a serializable error event."""
        ...
