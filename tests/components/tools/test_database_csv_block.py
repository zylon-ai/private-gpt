"""The CSV a truncated query result points at must be nameable to the model.

``LocalResourceBlock`` is the human's download attachment and never renders into
model-visible text, so ``_build_csv_block`` hands the canonical path back to the
caller, which names it in the sibling ``TextBlock``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.tools.builders.database_query_builder import (
    DatabaseQueryToolBuilder,
)
from private_gpt.events.models import BinaryBlock, LocalResourceBlock


def _builder(provider: str | None) -> DatabaseQueryToolBuilder:
    settings = MagicMock()
    settings.code_execution.provider = provider

    file_service = MagicMock()
    file_service.put_file = AsyncMock(
        return_value=MagicMock(id="file-1", mime_type="text/csv")
    )

    return DatabaseQueryToolBuilder(
        settings=settings,
        llm_component=MagicMock(),
        cache=MagicMock(),
        file_service=file_service,
    )


@pytest.mark.asyncio
async def test_stored_csv_returns_the_canonical_path() -> None:
    builder = _builder(provider="docker")

    block, path = await builder._build_csv_block(
        csv="a,b\n1,2\n", filename="result.csv", session_id="session-42"
    )

    assert isinstance(block, LocalResourceBlock)
    assert path == "/mnt/user-data/outputs/result.csv"
    assert block.file_path == path


@pytest.mark.asyncio
async def test_without_code_execution_there_is_no_path() -> None:
    builder = _builder(provider=None)

    block, path = await builder._build_csv_block(
        csv="a,b\n1,2\n", filename="result.csv", session_id="session-42"
    )

    assert isinstance(block, BinaryBlock)
    assert path is None


@pytest.mark.asyncio
async def test_without_a_session_there_is_no_path() -> None:
    builder = _builder(provider="docker")

    block, path = await builder._build_csv_block(
        csv="a,b\n1,2\n", filename="result.csv", session_id=None
    )

    assert isinstance(block, BinaryBlock)
    assert path is None


@pytest.mark.asyncio
async def test_a_failed_write_falls_back_to_a_blob_with_no_path() -> None:
    """The model must not be told to read a file that was never written."""
    builder = _builder(provider="docker")
    builder.file_service.put_file = AsyncMock(side_effect=RuntimeError("storage down"))

    block, path = await builder._build_csv_block(
        csv="a,b\n1,2\n", filename="result.csv", session_id="session-42"
    )

    assert isinstance(block, BinaryBlock)
    assert path is None
