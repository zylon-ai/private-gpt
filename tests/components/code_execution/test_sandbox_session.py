import unicodedata
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


# The stored name is pre-composed; the model sends it decomposed, as copied
# from search results. Both render identically.
_STORED = unicodedata.normalize("NFC", "/mnt/user-data/uploads/Memoria-Técnica.pdf")
_GIVEN = unicodedata.normalize("NFD", _STORED)


def _byte_exact_session(files: dict[str, bytes]) -> SandboxCodeExecutionSession:
    """A session over a filesystem that compares names byte-for-byte, like Linux."""

    async def path_exists(path: str) -> bool:
        return path in files

    async def read_file(path: str) -> bytes:
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    async def write_file(path: str, content: bytes) -> None:
        files[path] = content

    sandbox = SimpleNamespace(
        path_exists=AsyncMock(side_effect=path_exists),
        read_file=AsyncMock(side_effect=read_file),
        write_file=AsyncMock(side_effect=write_file),
        is_dir=AsyncMock(return_value=False),
    )
    env = SimpleNamespace(
        id="env-1",
        workspace="/home/agent/workspace/",
        sandbox=sandbox,
        touch=lambda: None,
        exec=AsyncMock(
            return_value=SimpleNamespace(
                success=True, stdout="", stderr="", exit_code=0, execution_time_ms=1
            )
        ),
    )
    return SandboxCodeExecutionSession(env)


def _executed(session: SandboxCodeExecutionSession) -> str:
    return session._env.exec.await_args.args[0]


@pytest.mark.asyncio
async def test_other_unicode_form_of_a_name_finds_the_file() -> None:
    session = _byte_exact_session({_STORED: b"%PDF"})

    assert _GIVEN != _STORED
    assert await session.path_exists(_GIVEN) is True
    assert await session.locate(_GIVEN) == _STORED
    assert await session.read_file(_GIVEN) == b"%PDF"


@pytest.mark.asyncio
async def test_locate_keeps_a_missing_path_as_given() -> None:
    session = _byte_exact_session({})

    assert await session.path_exists(_GIVEN) is False
    assert await session.locate(_GIVEN) == _GIVEN


@pytest.mark.asyncio
async def test_view_and_edit_reach_a_file_named_in_the_other_form() -> None:
    stored = unicodedata.normalize("NFC", "/home/agent/workspace/notas-café.md")
    files = {stored: b"alpha\n"}
    session = _byte_exact_session(files)
    given = unicodedata.normalize("NFD", stored)

    viewed = await session.view(given, include_line_numbers=False)
    replaced = await session.str_replace(given, "alpha", "beta")
    inserted = await session.insert(given, 1, "gamma")
    created = await session.create(given, "delta\n")

    assert viewed.success
    assert viewed.output == "alpha"
    assert replaced.success
    assert inserted.success
    assert created.success
    assert created.is_update
    # Every write landed on the existing file instead of creating a twin.
    assert list(files) == [stored]
    assert files[stored] == b"delta\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template",
    [
        "cp {path} /mnt/user-data/outputs/",
        "cp '{path}' /mnt/user-data/outputs/",
        'cp "{path}" /mnt/user-data/outputs/',
        "python -c 'open(\"{path}\")'",
        "INPUT={path} python run.py",
    ],
    ids=["bare", "single-quoted", "double-quoted", "nested", "assignment"],
)
async def test_bash_respells_paths_to_the_form_on_disk(template: str) -> None:
    session = _byte_exact_session({_STORED: b"%PDF"})

    await session.execute_bash(template.format(path=_GIVEN))

    assert template.format(path=_STORED) in _executed(session)


@pytest.mark.asyncio
async def test_bash_respells_paths_relative_to_the_workspace() -> None:
    stored = unicodedata.normalize("NFC", "/home/agent/workspace/informe-año.xlsx")
    session = _byte_exact_session({stored: b"PK"})

    await session.execute_bash(f"ls {unicodedata.normalize('NFD', 'informe-año.xlsx')}")

    assert f"ls {unicodedata.normalize('NFC', 'informe-año.xlsx')}" in _executed(
        session
    )


@pytest.mark.asyncio
async def test_bash_leaves_words_that_name_no_file_untouched() -> None:
    session = _byte_exact_session({})
    command = f"echo {unicodedata.normalize('NFD', 'café.txt')} > {_GIVEN}"

    await session.execute_bash(command)

    assert command in _executed(session)
    session._sandbox.path_exists.assert_awaited()


@pytest.mark.asyncio
async def test_bash_without_accented_paths_probes_nothing() -> None:
    session = _byte_exact_session({})

    await session.execute_bash("cp /mnt/user-data/uploads/report.pdf .")

    session._sandbox.path_exists.assert_not_awaited()
