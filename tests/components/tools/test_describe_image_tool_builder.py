import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.describe_image_tool_builder import (
    DescribeImageToolBuilder,
)

WORKSPACE = "/home/agent/workspace/"
UPLOADS = "/mnt/user-data/uploads/"

_DESCRIBE = (
    "private_gpt.components.tools.builders.describe_image_tool_builder.describe_image"
)


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


def _builder(
    session: _FakeSession | None, llm: object | None = None
) -> DescribeImageToolBuilder:
    component = SimpleNamespace(get_or_create_session=AsyncMock(return_value=session))
    model = llm if llm is not None else MagicMock()
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
    return DescribeImageToolBuilder(component, llm_component, settings)


def _config() -> CodeExecutionSessionConfig:
    return CodeExecutionSessionConfig(session_id="session-1")


async def _call(builder: DescribeImageToolBuilder, filepaths: list[str]):
    spec = await builder.build_tool(_config())
    assert spec.async_fn is not None
    return await spec.async_fn(filepaths=filepaths)


@pytest.mark.asyncio
async def test_describes_an_image_into_the_workspace() -> None:
    session = _FakeSession({f"{UPLOADS}chart.png": b"\x89PNG"})

    with patch(_DESCRIBE, AsyncMock(return_value="A bar chart.")) as described:
        blocks = await _call(_builder(session), [f"{UPLOADS}chart.png"])

    assert session.writes == [f"{WORKSPACE}chart.md"]
    assert session.files[f"{WORKSPACE}chart.md"] == b"A bar chart."
    assert f"{WORKSPACE}chart.md" in blocks[0].text
    assert blocks[-1].text == "Described 1 of 1 image(s)."

    # The raw bytes must reach the vision model, not a base64 re-encoding.
    block = described.await_args.args[1][0]
    assert block.resolve_image().read() == b"\x89PNG"
    assert block.image_mimetype == "image/png"


@pytest.mark.asyncio
async def test_a_second_description_of_the_same_stem_is_suffixed() -> None:
    session = _FakeSession({f"{UPLOADS}chart.png": b"\x89PNG"})
    builder = _builder(session)

    with patch(_DESCRIBE, AsyncMock(return_value="A bar chart.")):
        await _call(builder, [f"{UPLOADS}chart.png"])
        await _call(builder, [f"{UPLOADS}chart.png"])

    assert session.writes == [f"{WORKSPACE}chart.md", f"{WORKSPACE}chart (2).md"]


@pytest.mark.asyncio
async def test_same_stem_in_one_batch_does_not_race() -> None:
    session = _FakeSession(
        {f"{UPLOADS}chart.png": b"a", f"{UPLOADS}sub/chart.png": b"b"}
    )

    with patch(_DESCRIBE, AsyncMock(return_value="A chart.")):
        blocks = await _call(
            _builder(session), [f"{UPLOADS}chart.png", f"{UPLOADS}sub/chart.png"]
        )

    assert sorted(session.writes) == [
        f"{WORKSPACE}chart (2).md",
        f"{WORKSPACE}chart.md",
    ]
    assert blocks[-1].text == "Described 2 of 2 image(s)."


@pytest.mark.asyncio
async def test_images_are_described_in_parallel() -> None:
    session = _FakeSession({f"{UPLOADS}a.png": b"a", f"{UPLOADS}b.png": b"b"})
    in_flight = 0
    peak = 0

    async def _slow_describe(llm, blocks):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return "A picture."

    with patch(_DESCRIBE, _slow_describe):
        await _call(_builder(session), [f"{UPLOADS}a.png", f"{UPLOADS}b.png"])

    assert peak == 2


@pytest.mark.asyncio
async def test_a_bad_path_does_not_fail_its_siblings() -> None:
    session = _FakeSession({f"{UPLOADS}good.png": b"\x89PNG"})

    with patch(_DESCRIBE, AsyncMock(return_value="A picture.")):
        blocks = await _call(
            _builder(session), ["relative/path.png", f"{UPLOADS}good.png"]
        )

    assert "Error describing relative/path.png" in blocks[0].text
    assert f"{WORKSPACE}good.md" in blocks[1].text
    assert blocks[-1].text == "Described 1 of 2 image(s)."


@pytest.mark.asyncio
async def test_missing_file_reports_an_error_block() -> None:
    blocks = await _call(_builder(_FakeSession({})), [f"{UPLOADS}nope.png"])

    assert "File not found" in blocks[0].text
    assert blocks[-1].text == "Described 0 of 1 image(s)."


@pytest.mark.asyncio
async def test_empty_description_reports_an_error_block() -> None:
    session = _FakeSession({f"{UPLOADS}chart.png": b"\x89PNG"})

    with patch(_DESCRIBE, AsyncMock(return_value=None)):
        blocks = await _call(_builder(session), [f"{UPLOADS}chart.png"])

    assert "No description could be produced" in blocks[0].text
    assert session.writes == []


@pytest.mark.asyncio
async def test_no_image_capable_model_reports_an_error_block() -> None:
    session = _FakeSession({f"{UPLOADS}chart.png": b"\x89PNG"})
    component = SimpleNamespace(get_or_create_session=AsyncMock(return_value=session))
    llm_component = SimpleNamespace(
        get_llm=MagicMock(side_effect=ValueError("no models")),
        get_config=MagicMock(return_value=MagicMock()),
        filter=MagicMock(return_value=iter([])),
    )
    settings = SimpleNamespace(
        chat=SimpleNamespace(
            preprocess=SimpleNamespace(multimodal=SimpleNamespace(max_concurrency=8))
        )
    )
    builder = DescribeImageToolBuilder(component, llm_component, settings)

    with pytest.raises(ValueError, match="No image-capable model is configured"):
        await _call(builder, [f"{UPLOADS}chart.png"])


@pytest.mark.asyncio
async def test_no_provider_raises() -> None:
    with pytest.raises(ValueError, match="code_execution provider is not configured"):
        await _call(_builder(None), [f"{UPLOADS}chart.png"])


@pytest.mark.asyncio
async def test_empty_input_is_reported_not_crashed() -> None:
    blocks = await _call(_builder(_FakeSession({})), [])
    assert blocks[-1].text == "No images were requested."


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
        ":rebuild_describe_image_tool"
    )
    assert set(spec.execution_metadata.rebuild_kwargs) == {
        "config",
        "name",
        "type",
        "description",
    }
