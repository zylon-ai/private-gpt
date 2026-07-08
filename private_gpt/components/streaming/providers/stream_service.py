from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from private_gpt.components.streaming.providers.models import (
    StreamMetadata,
    StreamStatus,
)


class Event(NamedTuple):
    correlation_id: str
    event_data: str


class StreamService(ABC):
    """Abstract base class for streaming services."""

    @abstractmethod
    async def create_stream(
        self,
        stream_type: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new stream and return correlation ID."""
        pass

    @abstractmethod
    async def update_stream_status(
        self,
        correlation_id: str,
        status: StreamStatus,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update stream status and metadata."""
        pass

    @abstractmethod
    async def get_stream_metadata(self, correlation_id: str) -> StreamMetadata | None:
        """Get stream metadata by correlation ID."""
        pass

    @abstractmethod
    async def push_event(
        self,
        correlation_id: str,
        event_data: str,
    ) -> str:
        """Push raw event data to the stream."""
        pass

    @abstractmethod
    async def push_event_batch(
        self,
        events: list[Event],
    ) -> dict[str, str]:
        """Push multiple events to streams. Returns {correlation_id: last_event_id}."""
        pass

    @abstractmethod
    async def read_events(
        self,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[str], str]:
        """Read raw event data from stream."""
        pass

    @abstractmethod
    async def stream_exists(self, correlation_id: str) -> bool:
        """Check if stream exists."""
        pass

    @abstractmethod
    async def delete_stream(self, correlation_id: str) -> None:
        """Delete stream and its metadata."""
        pass

    @abstractmethod
    async def list_streams(
        self,
        stream_type: str | None = None,
        status: StreamStatus | None = None,
        limit: int | None = None,
    ) -> list[StreamMetadata]:
        """List streams with optional filtering."""
        pass

    @abstractmethod
    async def clean_up_stream(self, correlation_id: str) -> None:
        """Clean up resources associated with a stream."""
        pass

    @abstractmethod
    async def set_cancel_flag(self, correlation_id: str) -> None:
        """Set a cancellation flag for a stream.

        Used by the API process to signal a long-running chat worker that the
        stream should be cancelled. The worker polls :meth:`is_cancelled` inside
        its event loop.
        """
        pass

    @abstractmethod
    async def is_cancelled(self, correlation_id: str) -> bool:
        """Check whether a cancellation flag has been set for a stream."""
        pass

    @abstractmethod
    async def clear_cancel_flag(self, correlation_id: str) -> None:
        """Remove the cancellation flag for a stream (e.g. on cleanup)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass
