"""LocalObjectStorage must list POSIX-style relative paths.

Storage keys are POSIX strings by contract: the Files API service layer
splits them with ``path.split("/")``, matches them against ``uploads/`` /
``outputs/`` prefixes and feeds them to ``storage_to_canonical_path``, and the
S3 backend always returns ``/``-separated keys. ``LocalObjectStorage`` must
therefore also return ``/``-separated relative paths regardless of the host
OS, otherwise nested keys break on Windows (``data\\2024\\report.pdf``).
"""

import asyncio
from pathlib import Path

from private_gpt.components.storage.object_storage import LocalObjectStorage


def test_list_files_meta_returns_posix_paths_for_nested_keys(tmp_path: Path) -> None:
    storage = LocalObjectStorage(root_path=str(tmp_path))
    asyncio.run(
        storage.write_file(
            "uploads/session-1",
            "data/2024/report.pdf",
            b"pdf-bytes",
            "application/pdf",
        )
    )

    metas = asyncio.run(storage.list_files_meta("uploads/session-1"))

    assert [fi.path for fi in metas] == ["data/2024/report.pdf"]


def test_list_files_returns_posix_paths_for_nested_keys(tmp_path: Path) -> None:
    storage = LocalObjectStorage(root_path=str(tmp_path))
    asyncio.run(
        storage.write_file(
            "uploads/session-1",
            "data/2024/report.pdf",
            b"pdf-bytes",
            "application/pdf",
        )
    )

    rels = asyncio.run(storage.list_files("uploads/session-1"))

    assert rels == ["data/2024/report.pdf"]


def test_listed_paths_round_trip_through_read_file(tmp_path: Path) -> None:
    storage = LocalObjectStorage(root_path=str(tmp_path))
    asyncio.run(
        storage.write_file(
            "uploads/session-1",
            "data/2024/report.pdf",
            b"pdf-bytes",
            "application/pdf",
        )
    )

    metas = asyncio.run(storage.list_files_meta("uploads/session-1"))

    for meta in metas:
        content = asyncio.run(storage.read_file("uploads/session-1", meta.path))
        assert content == b"pdf-bytes"
