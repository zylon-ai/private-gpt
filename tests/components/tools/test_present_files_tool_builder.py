from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.tools.builders.present_files_tool_builder import (
    PresentFilesToolBuilder,
    _encode_file_id,
)
from private_gpt.events.models import LocalResourceBlock


@pytest.mark.asyncio
async def test_present_files_includes_file_etag() -> None:
    file_service = SimpleNamespace(
        head_file=AsyncMock(return_value=SimpleNamespace(etag='"version-1"'))
    )
    builder = PresentFilesToolBuilder(
        code_execution_component=SimpleNamespace(),
        file_service=file_service,
    )

    tool = await builder.build_tool(session_id="session-1")
    result = await tool.async_fn(filepaths=["/mnt/user-data/outputs/report.pdf"])

    resource = next(block for block in result if isinstance(block, LocalResourceBlock))
    assert resource.file_id == _encode_file_id("/mnt/user-data/outputs/report.pdf")
    assert resource.etag == '"version-1"'
    file_service.head_file.assert_awaited_once_with(
        scope_id="session-1",
        file_id=resource.file_id,
    )


@pytest.mark.asyncio
async def test_present_files_keeps_etag_optional_when_stat_fails() -> None:
    file_service = SimpleNamespace(
        head_file=AsyncMock(side_effect=RuntimeError("metadata unavailable"))
    )
    builder = PresentFilesToolBuilder(
        code_execution_component=SimpleNamespace(),
        file_service=file_service,
    )

    tool = await builder.build_tool(session_id="session-1")
    result = await tool.async_fn(filepaths=["/mnt/user-data/outputs/report.pdf"])

    resource = next(block for block in result if isinstance(block, LocalResourceBlock))
    assert resource.etag is None
    assert resource.model_dump() == {
        "type": "local_resource",
        "file_path": "/mnt/user-data/outputs/report.pdf",
        "file_id": _encode_file_id("/mnt/user-data/outputs/report.pdf"),
        "name": "report",
        "mime_type": "application/pdf",
    }
