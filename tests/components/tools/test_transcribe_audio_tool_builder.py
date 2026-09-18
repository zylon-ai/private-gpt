import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.transcribe_audio_tool_builder import (
    TranscribeAudioToolBuilder,
)

WORKSPACE = "/home/agent/workspace/"
UPLOADS = "/mnt/user-data/uploads/"

_TRANSCRIBE = (
    "private_gpt.components.tools.builders."
    "transcribe_audio_tool_builder.transcribe_audio"
)


def _result(transcript: str | None) -> SimpleNamespace:
    return SimpleNamespace(transcript=transcript)


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


def _builder(session: _FakeSession | None) -> TranscribeAudioToolBuilder:
    component = SimpleNamespace(get_or_create_session=AsyncMock(return_value=session))
    model = MagicMock()
    llm_component = SimpleNamespace(
        get_llm=MagicMock(return_value=model),
        get_config=MagicMock(return_value=MagicMock()),
        # A fresh iterator per call: the tool resolves the model on every
        # invocation, and a single exhausted one would look like no model.
        filter=MagicMock(side_effect=lambda _: iter([(model, MagicMock(), None)])),
    )
    settings = SimpleNamespace(
        chat=SimpleNamespace(
            preprocess=SimpleNamespace(multimodal=SimpleNamespace(max_concurrency=8))
        )
    )
    return TranscribeAudioToolBuilder(component, llm_component, settings)


def _config() -> CodeExecutionSessionConfig:
    return CodeExecutionSessionConfig(session_id="session-1")


async def _call(builder: TranscribeAudioToolBuilder, filepaths: list[str]):
    spec = await builder.build_tool(_config())
    assert spec.async_fn is not None
    return await spec.async_fn(filepaths=filepaths)


@pytest.mark.asyncio
async def test_transcribes_audio_into_the_workspace() -> None:
    session = _FakeSession({f"{UPLOADS}memo.mp3": b"ID3"})

    with patch(_TRANSCRIBE, AsyncMock(return_value=_result("Hello there."))) as called:
        blocks = await _call(_builder(session), [f"{UPLOADS}memo.mp3"])

    assert session.writes == [f"{WORKSPACE}memo.md"]
    assert session.files[f"{WORKSPACE}memo.md"] == b"Hello there."
    assert f"{WORKSPACE}memo.md" in blocks[0].text
    assert blocks[-1].text == "Transcribed 1 of 1 audio file(s)."

    # The raw bytes must reach the model, not a base64 re-encoding.
    block = called.await_args.args[1][0]
    assert block.resolve_audio().read() == b"ID3"
    assert block.format == "mp3"


@pytest.mark.asyncio
async def test_a_second_transcript_of_the_same_stem_is_suffixed() -> None:
    session = _FakeSession({f"{UPLOADS}memo.mp3": b"ID3"})
    builder = _builder(session)

    with patch(_TRANSCRIBE, AsyncMock(return_value=_result("Hello."))):
        await _call(builder, [f"{UPLOADS}memo.mp3"])
        await _call(builder, [f"{UPLOADS}memo.mp3"])

    assert session.writes == [f"{WORKSPACE}memo.md", f"{WORKSPACE}memo (2).md"]


@pytest.mark.asyncio
async def test_same_stem_in_one_batch_does_not_race() -> None:
    session = _FakeSession({f"{UPLOADS}memo.mp3": b"a", f"{UPLOADS}sub/memo.mp3": b"b"})

    with patch(_TRANSCRIBE, AsyncMock(return_value=_result("Hello."))):
        blocks = await _call(
            _builder(session), [f"{UPLOADS}memo.mp3", f"{UPLOADS}sub/memo.mp3"]
        )

    assert sorted(session.writes) == [f"{WORKSPACE}memo (2).md", f"{WORKSPACE}memo.md"]
    assert blocks[-1].text == "Transcribed 2 of 2 audio file(s)."


@pytest.mark.asyncio
async def test_audio_is_transcribed_in_parallel() -> None:
    session = _FakeSession({f"{UPLOADS}a.mp3": b"a", f"{UPLOADS}b.mp3": b"b"})
    in_flight = 0
    peak = 0

    async def _slow_transcribe(llm, blocks):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return _result("Hello.")

    with patch(_TRANSCRIBE, _slow_transcribe):
        await _call(_builder(session), [f"{UPLOADS}a.mp3", f"{UPLOADS}b.mp3"])

    assert peak == 2


@pytest.mark.asyncio
async def test_a_bad_path_does_not_fail_its_siblings() -> None:
    session = _FakeSession({f"{UPLOADS}good.mp3": b"ID3"})

    with patch(_TRANSCRIBE, AsyncMock(return_value=_result("Hello."))):
        blocks = await _call(
            _builder(session), ["relative/path.mp3", f"{UPLOADS}good.mp3"]
        )

    assert "Error transcribing relative/path.mp3" in blocks[0].text
    assert f"{WORKSPACE}good.md" in blocks[1].text
    assert blocks[-1].text == "Transcribed 1 of 2 audio file(s)."


@pytest.mark.asyncio
async def test_missing_file_reports_an_error_block() -> None:
    blocks = await _call(_builder(_FakeSession({})), [f"{UPLOADS}nope.mp3"])

    assert "File not found" in blocks[0].text
    assert blocks[-1].text == "Transcribed 0 of 1 audio file(s)."


@pytest.mark.asyncio
async def test_empty_transcript_reports_an_error_block() -> None:
    session = _FakeSession({f"{UPLOADS}memo.mp3": b"ID3"})

    with patch(_TRANSCRIBE, AsyncMock(return_value=_result(None))):
        blocks = await _call(_builder(session), [f"{UPLOADS}memo.mp3"])

    assert "No speech could be transcribed" in blocks[0].text
    assert session.writes == []


@pytest.mark.asyncio
async def test_no_provider_raises() -> None:
    with pytest.raises(ValueError, match="code_execution provider is not configured"):
        await _call(_builder(None), [f"{UPLOADS}memo.mp3"])


@pytest.mark.asyncio
async def test_empty_input_is_reported_not_crashed() -> None:
    blocks = await _call(_builder(_FakeSession({})), [])
    assert blocks[-1].text == "No audio files were requested."


def test_tool_schema_accepts_a_list_of_paths() -> None:
    spec = asyncio.run(_builder(_FakeSession({})).build_tool(_config()))

    properties = spec.input_schema["properties"]
    assert properties["filepaths"]["type"] == "array"
    assert properties["filepaths"]["items"]["type"] == "string"


def test_rebuild_metadata_is_set_for_celery() -> None:
    """Without this the tool silently no-ops under CeleryToolScheduler."""
    spec = asyncio.run(_builder(_FakeSession({})).build_tool(_config()))

    assert spec.execution_metadata is not None
    assert spec.execution_metadata.rebuild_callable.endswith(
        ":rebuild_transcribe_audio_tool"
    )
    assert set(spec.execution_metadata.rebuild_kwargs) == {
        "config",
        "name",
        "type",
        "description",
    }
