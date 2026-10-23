from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.code_execution.sandbox_session import (
    SandboxCodeExecutionSession,
)


def _session() -> SandboxCodeExecutionSession:
    sandbox = SimpleNamespace(
        path_exists=AsyncMock(),
        read_file=AsyncMock(),
        write_file=AsyncMock(),
    )
    return SandboxCodeExecutionSession(
        SimpleNamespace(id="env-1", workspace="/home/agent/workspace", sandbox=sandbox)
    )


@pytest.mark.asyncio
async def test_insert_rejects_missing_new_str() -> None:
    session = _session()

    result = await session.insert("notes.txt", 1, None)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "insert requires the new_str parameter."
    session._sandbox.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_rejects_missing_file_text() -> None:
    session = _session()

    result = await session.create("notes.txt", None)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "create requires the file_text parameter."
    session._sandbox.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_str_replace_rejects_missing_old_str() -> None:
    session = _session()

    result = await session.str_replace("notes.txt", None, "new")  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "str_replace requires the old_str parameter."
    session._sandbox.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_str_replace_rejects_missing_new_str() -> None:
    session = _session()

    result = await session.str_replace("notes.txt", "old", None)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error == "str_replace requires the new_str parameter."
    session._sandbox.read_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_file_delegates_to_sandbox() -> None:
    session = _session()
    session._env.touch = lambda: None

    await session.write_file("notes.txt", b"hello")

    session._sandbox.write_file.assert_awaited_once_with(
        "/home/agent/workspace/notes.txt", b"hello"
    )


def _viewing(content: bytes) -> SandboxCodeExecutionSession:
    session = _session()
    sandbox = session._sandbox
    sandbox.path_exists.return_value = True
    sandbox.is_dir = AsyncMock(return_value=False)
    sandbox.read_file.return_value = content
    session._env.touch = lambda: None
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00",
        b"PK\x03\x04\x14\x00\x06\x00",
        b"\xde\xad\xbe\xef\xfe\xed\xfa\xce",
    ],
    ids=["png", "jpeg", "zip", "invalid-utf8"],
)
async def test_view_rejects_a_binary_file(content: bytes) -> None:
    result = await _viewing(content).view("/mnt/user-data/uploads/blob")

    assert not result.success
    assert result.output is None or result.output == ""
    assert "binary" in (result.error or "")
    assert "/mnt/user-data/uploads/blob" in (result.error or "")


@pytest.mark.asyncio
async def test_view_still_opens_utf8_text() -> None:
    result = await _viewing("café\nline two\n".encode()).view(
        "/home/agent/workspace/notes.md", include_line_numbers=False
    )

    assert result.success
    assert result.output == "café\nline two"
    assert result.total_lines == 2


@pytest.mark.asyncio
async def test_view_opens_an_empty_file() -> None:
    result = await _viewing(b"").view("/home/agent/workspace/empty.txt")

    assert result.success
