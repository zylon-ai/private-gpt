import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.convert_documents_tool_builder import (
    ConvertDocumentsToolBuilder,
)

WORKSPACE = "/home/agent/workspace/"
UPLOADS = "/mnt/user-data/uploads/"


class _FakeSession:
    """In-memory stand-in for a sandbox session."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.writes: list[str] = []

    async def path_exists(self, path: str) -> bool:
        return path in self.files

    async def read_file(self, path: str) -> bytes:
        return self.files[path]

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content
        self.writes.append(path)


def _builder(session: _FakeSession | None, convert_service: object) -> tuple:
    component = SimpleNamespace(get_or_create_session=AsyncMock(return_value=session))
    scheduler_factory = SimpleNamespace(get=MagicMock(return_value=convert_service))
    settings = SimpleNamespace(
        chat=SimpleNamespace(
            preprocess=SimpleNamespace(documents=SimpleNamespace(max_concurrency=8))
        )
    )
    return ConvertDocumentsToolBuilder(component, scheduler_factory, settings)


def _config() -> CodeExecutionSessionConfig:
    return CodeExecutionSessionConfig(session_id="session-1")


async def _call(builder: ConvertDocumentsToolBuilder, filepaths: list[str]):
    spec = await builder.build_tool(_config())
    assert spec.async_fn is not None
    return await spec.async_fn(filepaths=filepaths)


@pytest.mark.asyncio
async def test_converts_a_document_into_the_workspace() -> None:
    session = _FakeSession({f"{UPLOADS}report.pdf": b"%PDF"})
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = "# Report"

    blocks = await _call(_builder(session, convert_service), [f"{UPLOADS}report.pdf"])

    assert session.writes == [f"{WORKSPACE}report.md"]
    assert session.files[f"{WORKSPACE}report.md"] == b"# Report"
    convert_service.bytes_to_text.assert_called_once_with(b"%PDF", ".pdf", False)
    assert f"{WORKSPACE}report.md" in blocks[0].text
    assert blocks[-1].text == "Converted 1 of 1 document(s)."


@pytest.mark.asyncio
async def test_a_second_conversion_of_the_same_stem_is_suffixed() -> None:
    session = _FakeSession({f"{UPLOADS}report.pdf": b"%PDF"})
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = "# Report"
    builder = _builder(session, convert_service)

    await _call(builder, [f"{UPLOADS}report.pdf"])
    await _call(builder, [f"{UPLOADS}report.pdf"])

    assert session.writes == [f"{WORKSPACE}report.md", f"{WORKSPACE}report (2).md"]


@pytest.mark.asyncio
async def test_same_stem_in_one_batch_does_not_race() -> None:
    """Parallel conversion must not let two documents claim the same name."""
    session = _FakeSession(
        {f"{UPLOADS}report.pdf": b"a", f"{UPLOADS}sub/report.pdf": b"b"}
    )
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = "# Report"

    blocks = await _call(
        _builder(session, convert_service),
        [f"{UPLOADS}report.pdf", f"{UPLOADS}sub/report.pdf"],
    )

    assert sorted(session.writes) == [
        f"{WORKSPACE}report (2).md",
        f"{WORKSPACE}report.md",
    ]
    assert blocks[-1].text == "Converted 2 of 2 document(s)."


@pytest.mark.asyncio
async def test_documents_convert_in_parallel() -> None:
    session = _FakeSession({f"{UPLOADS}a.pdf": b"a", f"{UPLOADS}b.pdf": b"b"})
    in_flight = 0
    peak = 0

    def _slow_convert(raw: bytes, ext: str, transform: bool) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Long enough that a sequential loop could not overlap the two calls.
        time.sleep(0.05)
        in_flight -= 1
        return "# Doc"

    convert_service = MagicMock()
    convert_service.bytes_to_text.side_effect = _slow_convert

    await _call(
        _builder(session, convert_service), [f"{UPLOADS}a.pdf", f"{UPLOADS}b.pdf"]
    )

    assert peak == 2


@pytest.mark.asyncio
async def test_a_bad_path_does_not_fail_its_siblings() -> None:
    session = _FakeSession({f"{UPLOADS}good.pdf": b"%PDF"})
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = "# Good"

    blocks = await _call(
        _builder(session, convert_service), ["relative/path.pdf", f"{UPLOADS}good.pdf"]
    )

    assert "Error converting relative/path.pdf" in blocks[0].text
    assert f"{WORKSPACE}good.md" in blocks[1].text
    assert blocks[-1].text == "Converted 1 of 2 document(s)."


@pytest.mark.asyncio
async def test_missing_file_reports_an_error_block() -> None:
    session = _FakeSession({})
    blocks = await _call(_builder(session, MagicMock()), [f"{UPLOADS}nope.pdf"])

    assert "File not found" in blocks[0].text
    assert blocks[-1].text == "Converted 0 of 1 document(s)."


@pytest.mark.asyncio
async def test_empty_conversion_reports_an_error_block() -> None:
    session = _FakeSession({f"{UPLOADS}report.pdf": b"%PDF"})
    convert_service = MagicMock()
    convert_service.bytes_to_text.return_value = ""

    blocks = await _call(_builder(session, convert_service), [f"{UPLOADS}report.pdf"])

    assert "No content could be extracted" in blocks[0].text
    assert session.writes == []


@pytest.mark.asyncio
async def test_no_provider_raises() -> None:
    with pytest.raises(ValueError, match="code_execution provider is not configured"):
        await _call(_builder(None, MagicMock()), [f"{UPLOADS}report.pdf"])


@pytest.mark.asyncio
async def test_empty_input_is_reported_not_crashed() -> None:
    blocks = await _call(_builder(_FakeSession({}), MagicMock()), [])
    assert blocks[-1].text == "No documents were requested."


def test_tool_schema_accepts_a_list_of_paths() -> None:
    builder = _builder(_FakeSession({}), MagicMock())
    spec = asyncio.run(builder.build_tool(_config()))

    properties = spec.input_schema["properties"]
    assert properties["filepaths"]["type"] == "array"
    assert properties["filepaths"]["items"]["type"] == "string"


def test_rebuild_metadata_is_set_for_celery() -> None:
    """Without this the tool silently no-ops under CeleryToolScheduler."""
    builder = _builder(_FakeSession({}), MagicMock())
    spec = asyncio.run(builder.build_tool(_config()))

    assert spec.execution_metadata is not None
    assert spec.execution_metadata.rebuild_callable.endswith(
        ":rebuild_convert_documents_tool"
    )
    assert set(spec.execution_metadata.rebuild_kwargs) == {
        "config",
        "name",
        "type",
        "description",
    }
