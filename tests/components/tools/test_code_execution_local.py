from pathlib import Path

import pytest

from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.code_execution.local import LocalCodeExecutionProvider
from private_gpt.settings.settings import unsafe_typed_settings


def _settings(tmp_path: Path):
    settings = unsafe_typed_settings.model_copy(deep=True)
    settings.code_execution.provider = "local"
    settings.code_execution.volume_root = None
    settings.code_execution.workspace_path = str(tmp_path / "workspaces")
    settings.code_execution.timeout = 5
    return settings


@pytest.mark.asyncio
async def test_local_code_execution_session_supports_file_operations(
    tmp_path: Path,
) -> None:
    provider = LocalCodeExecutionProvider(_settings(tmp_path))
    session = await provider.create_session(
        CodeExecutionSessionConfig(session_id="session-2")
    )

    created = await session.create("notes.txt", "alpha\nbeta\n")
    assert created.success is True

    view_all = await session.view("notes.txt")
    assert view_all.output == "1: alpha\n2: beta"

    replaced = await session.str_replace("notes.txt", "beta", "gamma")
    assert replaced.success is True

    inserted = await session.insert("notes.txt", 1, "between")
    assert inserted.success is True

    view_range = await session.view("notes.txt", (2, -1))
    assert view_range.output == "2: between\n3: gamma"

    listing = await session.view("/home/agent/workspace/")
    assert listing.success is True
    assert listing.output == "[file] notes.txt"

    await session.close()


@pytest.mark.asyncio
async def test_local_code_execution_session_rejects_unmounted_paths(
    tmp_path: Path,
) -> None:
    provider = LocalCodeExecutionProvider(_settings(tmp_path))
    session = await provider.create_session(
        CodeExecutionSessionConfig(session_id="session-3")
    )

    result = await session.create("/home/agent/../escape.txt", "nope")
    assert result.success is False
    assert "does not match any session mount" in (result.error or "")

    await session.close()


@pytest.mark.asyncio
async def test_local_code_execution_session_rejects_readonly_writes(
    tmp_path: Path,
) -> None:
    provider = LocalCodeExecutionProvider(_settings(tmp_path))
    session = await provider.create_session(
        CodeExecutionSessionConfig(session_id="session-4")
    )

    result = await session.create("/mnt/user-data/uploads/x.txt", "nope")
    assert result.success is False
    assert "read-only" in (result.error or "")

    await session.close()


@pytest.mark.asyncio
async def test_local_code_execution_files_survive_session_recreation(
    tmp_path: Path,
) -> None:
    provider = LocalCodeExecutionProvider(_settings(tmp_path))
    config = CodeExecutionSessionConfig(session_id="session-5")
    first = await provider.create_session(config)
    await first.create("keep.txt", "data")
    provider.delete_session(first)

    second = await provider.create_session(config)
    kept = await second.view("keep.txt")

    # The host directories outlive the sandbox — files reappear on reconnect.
    assert kept.success is True
    assert "data" in kept.output
