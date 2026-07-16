import datetime
import json
from typing import Any

import pytest

from private_gpt.events.models import (
    AudioBlock,
    BinaryBlock,
    ImageBlock,
    TextBlock,
    TLDRBlock,
    serialize_datetime,
)


@pytest.mark.parametrize(
    ("input_data", "expected_serialized", "description"),
    [
        # Direct instantiation - empty metadata excluded
        (
            {"content": []},
            {"type": "tldr", "tldr_side": "left", "content": []},
            "Empty metadata should be excluded from serialization",
        ),
        # Direct instantiation - non-empty metadata as _meta
        (
            {"metadata": {"pepe": "juan"}, "content": []},
            {
                "type": "tldr",
                "tldr_side": "left",
                "_meta": {"pepe": "juan"},
                "content": [],
            },
            "Non-empty metadata should serialize as _meta field",
        ),
        # Direct instantiation - empty dict metadata excluded
        (
            {"metadata": {}, "content": [TextBlock(text="test").model_dump()]},
            {
                "type": "tldr",
                "tldr_side": "left",
                "content": [{"type": "text", "text": "test"}],
            },
            "Empty metadata dict should be excluded",
        ),
        # Deserialization - from 'metadata' field
        (
            {"type": "tldr", "metadata": {"pepe": "juan"}, "content": []},
            {
                "type": "tldr",
                "tldr_side": "left",
                "_meta": {"pepe": "juan"},
                "content": [],
            },
            "Deserialize from 'metadata' field and serialize as '_meta'",
        ),
        # Deserialization - from '_meta' field
        (
            {"type": "tldr", "_meta": {"pepe": "juan"}, "content": []},
            {
                "type": "tldr",
                "tldr_side": "left",
                "_meta": {"pepe": "juan"},
                "content": [],
            },
            "Deserialize from '_meta' field and maintain as '_meta' with default tldr_side",
        ),
        # Complex metadata with multiple fields
        (
            {
                "metadata": {"author": "test", "version": "1.0", "tags": ["ai", "ml"]},
                "content": [
                    TextBlock(text="summary").model_dump(),
                    TextBlock(text="details").model_dump(),
                ],
            },
            {
                "type": "tldr",
                "tldr_side": "left",
                "_meta": {"author": "test", "version": "1.0", "tags": ["ai", "ml"]},
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "text", "text": "details"},
                ],
            },
            "Complex nested metadata should serialize correctly with tldr_side",
        ),
        # No metadata field at all - using proper TextBlock objects
        (
            {
                "type": "tldr",
                "content": [
                    TextBlock(text="item1").model_dump(),
                    TextBlock(text="item2").model_dump(),
                ],
            },
            {
                "type": "tldr",
                "tldr_side": "left",
                "content": [
                    {"type": "text", "text": "item1"},
                    {"type": "text", "text": "item2"},
                ],
            },
            "No metadata field should result in clean serialization with default tldr_side",
        ),
        # Metadata with nested structures
        (
            {
                "metadata": {
                    "config": {"enabled": True, "settings": {"level": 1}},
                    "timestamps": ["2024-01-01", "2024-01-02"],
                },
                "content": [],
            },
            {
                "type": "tldr",
                "tldr_side": "left",
                "_meta": {
                    "config": {"enabled": True, "settings": {"level": 1}},
                    "timestamps": ["2024-01-01", "2024-01-02"],
                },
                "content": [],
            },
            "Nested metadata structures should serialize correctly with tldr_side",
        ),
    ],
)
def test_tldr_block_metadata_serialization_deserialization(
    input_data: dict[str, Any], expected_serialized: dict[str, Any], description: str
) -> None:
    # Test object creation and model_dump
    block = TLDRBlock(**input_data)
    dict_result = block.model_dump()
    assert dict_result == expected_serialized, f"model_dump failed: {description}"

    # Test round-trip: serialize then deserialize using dict
    roundtrip_block = TLDRBlock(**dict_result)
    roundtrip_result = roundtrip_block.model_dump()
    assert roundtrip_result == expected_serialized, (
        f"Round-trip serialization failed: {description}"
    )

    # Test metadata field presence/absence in serialized output for TLDRBlock level
    input_metadata = input_data.get("metadata") or input_data.get("_meta")
    has_metadata = input_metadata is not None and bool(input_metadata)
    assert ("_meta" in dict_result) == has_metadata, (
        f"TLDRBlock metadata field presence assertion failed: {description}"
    )

    # Test that original metadata field is excluded from serialization
    assert "metadata" not in dict_result, (
        f"Original metadata field should be excluded: {description}"
    )

    # Test string representations don't raise exceptions
    str_repr = str(block)
    repr_repr = repr(block)
    assert isinstance(str_repr, str), f"String representation failed: {description}"
    assert isinstance(repr_repr, str), f"Repr representation failed: {description}"
    assert "TLDRBlock" in repr_repr, f"Repr should contain class name: {description}"


def test_image_block_source_base64_roundtrip() -> None:
    block = ImageBlock.model_validate(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "data": "aGVsbG8=",
                "media_type": "image/png",
            },
        }
    )

    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "image",
        "source": {
            "type": "base64",
            "data": "aGVsbG8=",
            "media_type": "image/png",
        },
    }
    assert block.source.data == "aGVsbG8="
    assert block.source.media_type == "image/png"


def test_audio_block_source_base64_roundtrip() -> None:
    block = AudioBlock.model_validate(
        {
            "type": "audio",
            "source": {
                "type": "base64",
                "data": "YXVkaW8=",
                "media_type": "audio/mpeg",
            },
        }
    )

    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "audio",
        "source": {
            "type": "base64",
            "data": "YXVkaW8=",
            "media_type": "audio/mpeg",
        },
    }
    assert block.source.data == "YXVkaW8="
    assert block.source.media_type == "audio/mpeg"


def test_binary_block_source_base64_roundtrip() -> None:
    block = BinaryBlock.model_validate(
        {
            "type": "binary",
            "filename": "document.pdf",
            "source": {
                "type": "base64",
                "data": "YmluYXJ5",
                "media_type": "application/pdf",
            },
        }
    )

    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "binary",
        "filename": "document.pdf",
        "source": {
            "type": "base64",
            "data": "YmluYXJ5",
            "media_type": "application/pdf",
        },
    }
    assert block.source.type == "base64"
    assert block.source.data == "YmluYXJ5"
    assert block.source.media_type == "application/pdf"


def test_image_block_legacy_payload_is_still_accepted() -> None:
    block = ImageBlock.model_validate(
        {"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"]["type"] == "base64"


def test_audio_block_legacy_payload_is_still_accepted() -> None:
    block = AudioBlock.model_validate(
        {"type": "audio", "data": "YXVkaW8=", "mime_type": "audio/mpeg"}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"]["type"] == "base64"


def test_binary_block_legacy_payload_is_still_accepted() -> None:
    block = BinaryBlock.model_validate(
        {
            "type": "binary",
            "data": "YmluYXJ5",
            "mime_type": "application/pdf",
        }
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"]["type"] == "base64"


def test_image_block_legacy_url_is_still_accepted() -> None:
    block = ImageBlock.model_validate({"type": "image", "url": "s3://bucket/image.png"})
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"] == {"type": "url", "url": "s3://bucket/image.png"}


def test_audio_block_legacy_url_is_still_accepted() -> None:
    block = AudioBlock.model_validate({"type": "audio", "url": "s3://bucket/audio.mp3"})
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"] == {"type": "url", "url": "s3://bucket/audio.mp3"}


def test_binary_block_legacy_url_is_still_accepted() -> None:
    block = BinaryBlock.model_validate(
        {"type": "binary", "url": "s3://bucket/file.pdf"}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped["source"] == {"type": "url", "url": "s3://bucket/file.pdf"}


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/i.png",
        "http://example.com/i.png",
        "s3://bucket/image.png",
        "ftp://example.com/i.png",
        "custom+v1://bucket/image.png",
    ],
)
def test_image_block_source_url_roundtrip(url: str) -> None:
    block = ImageBlock.model_validate(
        {"type": "image", "source": {"type": "url", "url": url}}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "image",
        "source": {"type": "url", "url": url},
    }
    assert block.source.url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a.mp3",
        "http://example.com/a.mp3",
        "s3://bucket/audio.mp3",
        "ftp://example.com/a.mp3",
        "custom+v1://bucket/audio.mp3",
    ],
)
def test_audio_block_source_url_roundtrip(url: str) -> None:
    block = AudioBlock.model_validate(
        {"type": "audio", "source": {"type": "url", "url": url}}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "audio",
        "source": {"type": "url", "url": url},
    }
    assert block.source.url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/f.pdf",
        "http://example.com/f.pdf",
        "s3://bucket/file.pdf",
        "ftp://example.com/f.pdf",
        "custom+v1://bucket/file.pdf",
    ],
)
def test_binary_block_source_uri_roundtrip(url: str) -> None:
    block = BinaryBlock.model_validate(
        {"type": "binary", "source": {"type": "url", "url": url}}
    )
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {
        "type": "binary",
        "source": {"type": "url", "url": url},
    }
    assert block.source.type == "url"
    assert block.source.url == url


@pytest.mark.parametrize(
    ("input_data", "expected_serialized", "description"),
    [
        # No timestamps - should be excluded from serialization
        (
            {"type": "text", "text": "sample"},
            {"type": "text", "text": "sample"},
            "None timestamps should be excluded from serialization",
        ),
        # Only start_timestamp set
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(2024, 1, 15, 10, 30, 0),
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00Z",
            },
            "Only start_timestamp should serialize to ISO format",
        ),
        # Only stop_timestamp set
        (
            {
                "type": "text",
                "text": "sample",
                "stop_timestamp": datetime.datetime(2024, 1, 15, 11, 45, 30),
            },
            {
                "type": "text",
                "text": "sample",
                "stop_timestamp": "2024-01-15T11:45:30Z",
            },
            "Only stop_timestamp should serialize to ISO format",
        ),
        # Both timestamps set
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(2024, 1, 15, 10, 30, 0),
                "stop_timestamp": datetime.datetime(2024, 1, 15, 11, 45, 30),
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00Z",
                "stop_timestamp": "2024-01-15T11:45:30Z",
            },
            "Both timestamps should serialize to ISO format",
        ),
        # Timestamps with microseconds
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(2024, 1, 15, 10, 30, 0, 123456),
                "stop_timestamp": datetime.datetime(2024, 1, 15, 11, 45, 30, 987654),
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00.123456Z",
                "stop_timestamp": "2024-01-15T11:45:30.987654Z",
            },
            "Timestamps with microseconds should serialize correctly",
        ),
        # Deserialization from ISO strings
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00",
                "stop_timestamp": "2024-01-15T11:45:30",
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00Z",
                "stop_timestamp": "2024-01-15T11:45:30Z",
            },
            "Deserialization from ISO strings should work correctly",
        ),
        # Mixed explicit None values (should be excluded)
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(2024, 1, 15, 10, 30, 0),
                "stop_timestamp": None,
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00Z",
            },
            "Explicit None timestamp should be excluded from serialization",
        ),
        # Timezone-aware datetimes (converted to UTC)
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(
                    2024, 1, 15, 10, 30, 0, tzinfo=datetime.UTC
                ),
                "stop_timestamp": datetime.datetime(
                    2024,
                    1,
                    15,
                    11,
                    45,
                    30,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=5)),
                ),
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T10:30:00Z",
                "stop_timestamp": "2024-01-15T06:45:30Z",  # Converted from UTC+5 to UTC
            },
            "Timezone-aware datetimes should be converted to UTC",
        ),
        # With metadata and timestamps
        (
            {
                "type": "text",
                "text": "sample",
                "metadata": {"source": "test"},
                "start_timestamp": datetime.datetime(2024, 1, 15, 10, 30, 0),
                "stop_timestamp": datetime.datetime(2024, 1, 15, 11, 45, 30),
            },
            {
                "type": "text",
                "text": "sample",
                "_meta": {"source": "test"},
                "start_timestamp": "2024-01-15T10:30:00Z",
                "stop_timestamp": "2024-01-15T11:45:30Z",
            },
            "Timestamps with metadata should serialize correctly",
        ),
        # Test different timezone offsets to ensure UTC conversion
        (
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": datetime.datetime(
                    2024,
                    1,
                    15,
                    10,
                    30,
                    0,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=-5)),  # EST
                ),
                "stop_timestamp": datetime.datetime(
                    2024,
                    1,
                    15,
                    10,
                    30,
                    0,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=9)),  # JST
                ),
            },
            {
                "type": "text",
                "text": "sample",
                "start_timestamp": "2024-01-15T15:30:00Z",  # EST to UTC
                "stop_timestamp": "2024-01-15T01:30:00Z",  # JST to UTC
            },
            "Various timezone offsets should convert to UTC correctly",
        ),
    ],
)
def test_datetime_serialization_deserialization(
    input_data: dict[str, Any], expected_serialized: dict[str, Any], description: str
) -> None:
    """Test datetime field serialization and deserialization for TextBlock."""
    # Test object creation and model_dump
    block = TextBlock(**input_data)
    dict_result = json.loads(block.model_dump_json())
    assert dict_result == expected_serialized, f"model_dump failed: {description}"

    # Test round-trip: serialize then deserialize using dict
    roundtrip_block = TextBlock(**dict_result)
    roundtrip_result = json.loads(roundtrip_block.model_dump_json())
    assert roundtrip_result == expected_serialized, (
        f"Round-trip serialization failed: {description}"
    )

    # Test timestamp field presence/absence in serialized output
    input_start = input_data.get("start_timestamp")
    input_stop = input_data.get("stop_timestamp")

    has_start = input_start is not None
    has_stop = input_stop is not None

    assert ("start_timestamp" in dict_result) == has_start, (
        f"start_timestamp field presence assertion failed: {description}"
    )

    assert ("stop_timestamp" in dict_result) == has_stop, (
        f"stop_timestamp field presence assertion failed: {description}"
    )

    # Test that datetime objects are properly converted to strings in serialization
    if has_start and isinstance(input_start, datetime.datetime):
        assert isinstance(dict_result["start_timestamp"], str), (
            f"start_timestamp should be serialized as string: {description}"
        )

    if has_stop and isinstance(input_stop, datetime.datetime):
        assert isinstance(dict_result["stop_timestamp"], str), (
            f"stop_timestamp should be serialized as string: {description}"
        )

    # Test that deserialized objects have proper datetime types
    if has_start:
        assert isinstance(block.start_timestamp, datetime.datetime), (
            f"Deserialized start_timestamp should be datetime object: {description}"
        )

    if has_stop:
        assert isinstance(block.stop_timestamp, datetime.datetime), (
            f"Deserialized stop_timestamp should be datetime object: {description}"
        )

    # Test string representations don't raise exceptions
    str_repr = str(block)
    repr_repr = repr(block)
    assert isinstance(str_repr, str), f"String representation failed: {description}"
    assert isinstance(repr_repr, str), f"Repr representation failed: {description}"
    assert "TextBlock" in repr_repr, f"Repr should contain class name: {description}"


@pytest.mark.parametrize(
    ("timezone_offset", "expected_result"),
    [
        (0, "2024-01-15T10:30:00Z"),  # UTC
        (-5, "2024-01-15T10:30:00Z"),  # EST (naive treated as UTC)
        (1, "2024-01-15T10:30:00Z"),  # CET (naive treated as UTC)
        (9, "2024-01-15T10:30:00Z"),  # JST (naive treated as UTC)
    ],
)
def test_environment_independence(timezone_offset: int, expected_result: str) -> None:
    naive_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
    result = serialize_datetime(naive_dt)
    assert result == expected_result, (
        f"Environment independence failed for UTC{timezone_offset:+d}"
    )


@pytest.mark.parametrize(
    ("input_dt", "expected_result", "description"),
    [
        (
            datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=datetime.UTC),
            "2024-01-15T10:30:00Z",
            "UTC timezone should remain UTC",
        ),
        (
            datetime.datetime(
                2024,
                1,
                15,
                10,
                30,
                0,
                tzinfo=datetime.timezone(datetime.timedelta(hours=5)),
            ),
            "2024-01-15T05:30:00Z",
            "UTC+5 should convert to UTC",
        ),
        (
            datetime.datetime(
                2024,
                1,
                15,
                10,
                30,
                0,
                tzinfo=datetime.timezone(datetime.timedelta(hours=-8)),
            ),
            "2024-01-15T18:30:00Z",
            "UTC-8 should convert to UTC",
        ),
        (
            datetime.datetime(
                2024,
                1,
                15,
                15,
                45,
                30,
                tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
            ),
            "2024-01-15T13:45:30Z",
            "UTC+2 should convert to UTC",
        ),
        (
            datetime.datetime(
                2024,
                1,
                15,
                5,
                15,
                0,
                tzinfo=datetime.timezone(datetime.timedelta(hours=-10)),
            ),
            "2024-01-15T15:15:00Z",
            "UTC-10 should convert to UTC",
        ),
    ],
)
def test_timezone_aware_conversion(
    input_dt: datetime.datetime, expected_result: str, description: str
) -> None:
    """Test that timezone-aware datetimes are properly converted to UTC."""
    result = serialize_datetime(input_dt)
    assert result == expected_result, f"Timezone conversion failed: {description}"
