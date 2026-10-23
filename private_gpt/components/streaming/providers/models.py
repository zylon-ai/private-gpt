import enum
import json
import uuid
from datetime import UTC, datetime
from functools import total_ordering
from typing import Any, Union

from pydantic import BaseModel, Field


@total_ordering
class StreamStatus(enum.StrEnum):
    """Enumeration of possible stream statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"

    def _get_order(self) -> int:
        return list(StreamStatus).index(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StreamStatus):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Union["StreamStatus", object]) -> bool:
        if not isinstance(other, StreamStatus):
            return NotImplemented
        return self._get_order() < other._get_order()

    def __le__(self, other: Union["StreamStatus", object]) -> bool:
        if not isinstance(other, StreamStatus):
            return NotImplemented
        return self._get_order() <= other._get_order()

    def __gt__(self, other: Union["StreamStatus", object]) -> bool:
        if not isinstance(other, StreamStatus):
            return NotImplemented
        return self._get_order() > other._get_order()

    def __ge__(self, other: Union["StreamStatus", object]) -> bool:
        if not isinstance(other, StreamStatus):
            return NotImplemented
        return self._get_order() >= other._get_order()

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        """Return the string representation of the status."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "StreamStatus":
        """Create StreamStatus from string value."""
        normalized = str(value).lower().strip()
        for status in cls:
            if status.value == normalized:
                return status
        raise ValueError(
            f"Invalid status: {value}. Valid options: {[s.value for s in cls]}"
        )


class StreamMetadata(BaseModel):
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the stream",
    )
    status: StreamStatus = Field(
        default=StreamStatus.PENDING,
        description="Current status of the stream",
    )
    created_at: datetime = Field(
        default=datetime.now(UTC),
        description="Timestamp when the stream was created",
    )
    updated_at: datetime = Field(
        default=datetime.now(UTC),
        description="Timestamp when the stream was last updated",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the stream was completed, if applicable",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the stream encountered an error",
    )
    stream_type: str = Field(
        default="default",
        description="Type of the stream, used for categorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the stream",
    )

    def model_dump_json_fields(self) -> dict[str, str]:
        data = self.model_dump()
        return {
            k: v.model_dump_json()
            if isinstance(v, BaseModel)
            else json.dumps(v)
            if isinstance(v, dict | list)
            else v.isoformat()
            if isinstance(v, datetime)
            else str(v)
            if v is not None
            else ""
            for k, v in data.items()
        }
