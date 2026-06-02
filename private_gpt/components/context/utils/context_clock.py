"""Provide context clock utilities."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return an aware current UTC timestamp."""
    return datetime.now(UTC)
