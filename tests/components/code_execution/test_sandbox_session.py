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
