"""Tests for the file event debouncer and MinIO notification parser."""

from __future__ import annotations

from private_gpt.components.filesystems.callbacks.file_watcher import (
    FileEventDebouncer,
    _is_temp_key,
    parse_minio_notification,
)


class TestTempKeyDetection:
    def test_minio_sys_prefix_is_temp(self) -> None:
        assert _is_temp_key(".minio.sys/tmp/upload")

    def test_tmp_suffix_is_temp(self) -> None:
        assert _is_temp_key("org/art.tmp")

    def test_part_suffix_is_temp(self) -> None:
        assert _is_temp_key("org/upload.part")

    def test_normal_key_is_not_temp(self) -> None:
        assert not _is_temp_key("org-1/proj-1/art-1.mdx")

    def test_tilde_suffix_is_temp(self) -> None:
        assert _is_temp_key("org/file~")


class TestMinioNotificationParser:
    def _make_payload(
        self,
        key: str = "org-1/art-1.mdx",
        event: str = "s3:ObjectCreated:Put",
        size: int = 1024,
    ) -> dict:
        return {
            "Records": [
                {
                    "eventName": event,
                    "s3": {
                        "bucket": {"name": "artifacts"},
                        "object": {"key": key, "size": size},
                    },
                }
            ]
        }

    def test_created_event_parsed(self) -> None:
        results = parse_minio_notification(self._make_payload())
        assert len(results) == 1
        key, event_type, size = results[0]
        assert event_type == "file.created"
        assert key == "org-1/art-1.mdx"
        assert size == 1024

    def test_deleted_event_parsed(self) -> None:
        results = parse_minio_notification(
            self._make_payload(event="s3:ObjectRemoved:Delete")
        )
        assert len(results) == 1
        _, event_type, _ = results[0]
        assert event_type == "file.deleted"

    def test_zero_byte_created_skipped(self) -> None:
        results = parse_minio_notification(self._make_payload(size=0))
        assert results == []

    def test_temp_key_skipped(self) -> None:
        results = parse_minio_notification(
            self._make_payload(key="org/.minio.sys/upload")
        )
        assert results == []

    def test_unknown_event_skipped(self) -> None:
        results = parse_minio_notification(
            self._make_payload(event="s3:Replication:OperationCompletedReplication")
        )
        assert results == []

    def test_empty_records(self) -> None:
        assert parse_minio_notification({"Records": []}) == []


class TestFileEventDebouncer:
    def test_first_event_emitted(self) -> None:
        debouncer = FileEventDebouncer()
        assert debouncer.should_emit("org/art.mdx", "file.created") is True

    def test_immediate_duplicate_suppressed(self) -> None:
        debouncer = FileEventDebouncer()
        debouncer.should_emit("org/art.mdx", "file.created")
        # Immediate duplicate within debounce window
        assert debouncer.should_emit("org/art.mdx", "file.created") is False

    def test_different_event_types_not_suppressed(self) -> None:
        debouncer = FileEventDebouncer()
        debouncer.should_emit("org/art.mdx", "file.created")
        # Different type is not a dup
        assert debouncer.should_emit("org/art.mdx", "file.deleted") is True

    def test_different_keys_not_suppressed(self) -> None:
        debouncer = FileEventDebouncer()
        debouncer.should_emit("org/art-1.mdx", "file.created")
        assert debouncer.should_emit("org/art-2.mdx", "file.created") is True
