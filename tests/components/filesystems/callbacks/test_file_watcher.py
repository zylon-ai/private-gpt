"""Tests for the file watcher deduplication logic (T5.1)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from private_gpt.components.filesystems.callbacks.file_watcher import (
    _DebouncedHandler,
    _is_temp_file,
)
from private_gpt.components.filesystems.callbacks.models import FileEvent


def _make_handler(
    tmp_path: Path,
    collected: list[FileEvent],
    loop: asyncio.AbstractEventLoop,
) -> _DebouncedHandler:
    async def emit(ev: FileEvent) -> None:
        collected.append(ev)

    return _DebouncedHandler(
        namespace="artifacts",
        scope="org-1",
        root=tmp_path,
        correlation={"turn": "t1"},
        emit=emit,
        loop=loop,
    )


class TestTempFileDetection:
    def test_tilde_prefix_is_temp(self) -> None:
        assert _is_temp_file("~doc.md")

    def test_tmp_suffix_is_temp(self) -> None:
        assert _is_temp_file("file.tmp")

    def test_part_suffix_is_temp(self) -> None:
        assert _is_temp_file("upload.part")

    def test_crdownload_is_temp(self) -> None:
        assert _is_temp_file("file.crdownload")

    def test_normal_file_is_not_temp(self) -> None:
        assert not _is_temp_file("document.mdx")

    def test_hidden_temp_is_temp(self) -> None:
        assert _is_temp_file(".~lock.doc")


class TestDebouncedHandler:
    @pytest.fixture
    def loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_single_write_emits_one_event(
        self, tmp_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        collected: list[FileEvent] = []
        handler = _make_handler(tmp_path, collected, loop)
        # Create a real file so size check passes
        target = tmp_path / "result.mdx"
        target.write_bytes(b"content")
        handler._handle(str(target), "file.created")
        loop.run_until_complete(asyncio.sleep(0))
        assert len(collected) == 1
        assert collected[0].type == "file.created"
        assert collected[0].namespace == "artifacts"
        assert collected[0].correlation == {"turn": "t1"}

    def test_rapid_duplicate_events_collapsed(
        self, tmp_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        collected: list[FileEvent] = []
        handler = _make_handler(tmp_path, collected, loop)
        target = tmp_path / "report.mdx"
        target.write_bytes(b"some content here")
        # Fire the same event rapidly
        handler._handle(str(target), "file.updated")
        handler._handle(str(target), "file.updated")
        handler._handle(str(target), "file.updated")
        loop.run_until_complete(asyncio.sleep(0))
        assert len(collected) == 1

    def test_empty_file_skipped_for_create(
        self, tmp_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        collected: list[FileEvent] = []
        handler = _make_handler(tmp_path, collected, loop)
        empty = tmp_path / "placeholder.tmp"
        empty.write_bytes(b"")
        handler._handle(str(empty), "file.created")
        loop.run_until_complete(asyncio.sleep(0))
        assert collected == []

    def test_delete_event_fires_even_for_gone_file(
        self, tmp_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        collected: list[FileEvent] = []
        handler = _make_handler(tmp_path, collected, loop)
        gone = tmp_path / "deleted.mdx"
        # File doesn't exist — delete events should still fire
        handler._handle(str(gone), "file.deleted")
        loop.run_until_complete(asyncio.sleep(0))
        assert len(collected) == 1
        assert collected[0].type == "file.deleted"

    def test_temp_file_event_skipped(
        self, tmp_path: Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        collected: list[FileEvent] = []
        handler = _make_handler(tmp_path, collected, loop)
        temp = tmp_path / "upload.part"
        temp.write_bytes(b"partial content")
        handler._handle(str(temp), "file.created")
        loop.run_until_complete(asyncio.sleep(0))
        assert collected == []
