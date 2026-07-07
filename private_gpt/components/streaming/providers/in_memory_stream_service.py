import asyncio
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from private_gpt.components.streaming.providers.models import (
    StreamMetadata,
    StreamStatus,
)
from private_gpt.components.streaming.providers.stream_service import (
    Event,
    StreamService,
)


class InMemoryStreamService(StreamService):
    """Simple in-memory implementation that mimics Redis behavior exactly."""

    def __init__(self) -> None:
        # Mimic Redis: separate storage for metadata (hash) and events (stream)
        self._metadata: dict[str, StreamMetadata] = {}  # Like Redis hash
        self._events: dict[
            str, list[tuple[str, str]]
        ] = {}  # Like Redis stream: [(id, data)]
        self._event_counters: dict[str, int] = {}
        self._lock = asyncio.Lock()
        # Per-stream waiters: readers park here instead of sleeping
        self._waiters: dict[str, set[asyncio.Event]] = defaultdict(set)

    async def create_stream(
        self,
        stream_type: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new stream and return correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        elif correlation_id and await self.stream_exists(correlation_id):
            raise ValueError(
                f"Stream with correlation_id {correlation_id} already exists"
            )

        now = datetime.now(UTC)
        stream_metadata = StreamMetadata(
            correlation_id=correlation_id,
            status=StreamStatus.PENDING,
            created_at=now,
            updated_at=now,
            stream_type=stream_type,
            metadata=metadata or {},
        )

        async with self._lock:
            self._metadata[correlation_id] = stream_metadata
            self._events[correlation_id] = []
            self._event_counters[correlation_id] = 0

        return correlation_id

    async def update_stream_status(
        self,
        correlation_id: str,
        status: StreamStatus,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update stream status and metadata."""
        async with self._lock:
            if correlation_id not in self._metadata:
                raise ValueError(
                    f"Stream with correlation_id {correlation_id} not found"
                )

            stream_meta = self._metadata[correlation_id]
            stream_meta.status = status
            stream_meta.updated_at = datetime.now(UTC)

            if error_message:
                stream_meta.error_message = error_message

            if status in [
                StreamStatus.COMPLETED,
                StreamStatus.CANCELLED,
                StreamStatus.ERROR,
            ]:
                stream_meta.completed_at = datetime.now(UTC)

            if metadata:
                stream_meta.metadata.update(metadata)

    async def get_stream_metadata(self, correlation_id: str) -> StreamMetadata | None:
        """Get stream metadata by correlation ID."""
        async with self._lock:
            return self._metadata.get(correlation_id)

    async def push_event(self, correlation_id: str, event_data: str) -> str:
        """Push raw event data to the stream."""
        async with self._lock:
            if correlation_id not in self._metadata:
                raise ValueError(
                    f"Stream with correlation_id {correlation_id} not found"
                )

            # Generate sequential message ID like Redis
            self._event_counters[correlation_id] += 1
            message_id = f"{int(datetime.now(UTC).timestamp() * 1000)}-{self._event_counters[correlation_id]}"

            self._events[correlation_id].append((message_id, event_data))
            for waiter in list(self._waiters.get(correlation_id, set())):
                waiter.set()
            return message_id

    async def push_event_batch(self, events: list[Event]) -> dict[str, str]:
        for event in events:
            if event.correlation_id not in self._metadata:
                raise ValueError(
                    f"Stream with correlation_id {event.correlation_id} not found"
                )
        grouped: dict[str, list[str]] = defaultdict(list)
        for event in events:
            grouped[event.correlation_id].append(event.event_data)

        result: dict[str, str] = {}
        for correlation_id, event_datas in grouped.items():
            last_id = None
            for event_data in event_datas:
                last_id = await self.push_event(correlation_id, event_data)
            if last_id:
                result[correlation_id] = last_id
        return result

    async def read_events(
        self,
        correlation_id: str,
        last_id: str = "0",
        count: int | None = None,
        block_ms: int | None = None,
    ) -> tuple[list[str], str]:
        """Read raw event data from stream and return next last_id."""

        # Non-blocking read
        def _read_available() -> tuple[list[str], str]:
            if correlation_id not in self._events:
                return [], last_id

            events = self._events[correlation_id]

            # Find events after last_id
            start_idx = 0
            if last_id != "0":
                for i, (msg_id, _) in enumerate(events):
                    if msg_id == last_id:
                        start_idx = i + 1
                        break

            # Get events from start_idx
            available_events = events[start_idx:]
            if count is not None:
                available_events = available_events[:count]

            event_data = [data for _, data in available_events]
            next_last_id = available_events[-1][0] if available_events else last_id

            return event_data, next_last_id

        waiter: asyncio.Event | None = None
        async with self._lock:
            event_data, next_last_id = _read_available()
            if not event_data and block_ms:
                # Register waiter atomically with the "no events" check so we
                # cannot miss a push_event that fires between check and wait.
                waiter = asyncio.Event()
                self._waiters[correlation_id].add(waiter)

        if waiter is not None:
            assert block_ms is not None
            try:
                await asyncio.wait_for(waiter.wait(), timeout=block_ms / 1000)
            except TimeoutError:
                pass
            finally:
                async with self._lock:
                    self._waiters[correlation_id].discard(waiter)
            async with self._lock:
                event_data, next_last_id = _read_available()

        return event_data, next_last_id

    async def stream_exists(self, correlation_id: str) -> bool:
        """Check if stream exists."""
        async with self._lock:
            return correlation_id in self._metadata

    async def delete_stream(self, correlation_id: str) -> None:
        """Delete stream and its metadata."""
        async with self._lock:
            if correlation_id not in self._metadata:
                raise ValueError(
                    f"Stream with correlation_id {correlation_id} not found"
                )

            del self._metadata[correlation_id]
            del self._events[correlation_id]
            del self._event_counters[correlation_id]
            for waiter in self._waiters.pop(correlation_id, set()):
                waiter.set()

    async def list_streams(
        self,
        stream_type: str | None = None,
        status: StreamStatus | None = None,
        limit: int | None = None,
    ) -> list[StreamMetadata]:
        """List streams with optional filtering."""
        async with self._lock:
            streams = list(self._metadata.values())

            # Apply filters
            if stream_type:
                streams = [s for s in streams if s.stream_type == stream_type]
            if status:
                streams = [s for s in streams if s.status == status]

            # Sort by created_at (newest first)
            streams.sort(key=lambda x: x.created_at, reverse=True)

            # Apply limit
            if limit:
                streams = streams[:limit]

            return streams

    async def clean_up_stream(self, correlation_id: str) -> None:
        """Clean up a stream by deleting it and its events."""
        await self.delete_stream(correlation_id)

    async def close(self) -> None:
        """Close any open connections and clean up resources."""
        async with self._lock:
            for waiters in self._waiters.values():
                for waiter in waiters:
                    waiter.set()
            self._waiters.clear()
            self._metadata.clear()
            self._events.clear()
            self._event_counters.clear()
