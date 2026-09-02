from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from private_gpt.components.sandbox.base import SandboxSession
from private_gpt.components.tabular.pandasai_sandbox import PandasAISandboxAdapter

if TYPE_CHECKING:
    from pathlib import Path


def test_stop_force_closes_session_and_removes_temporary_directory(tmp_path: Path) -> None:
    client = AsyncMock(spec=SandboxSession)
    adapter = PandasAISandboxAdapter(client=client)
    adapter._started = True
    adapter._temp_dir = tmp_path / "sandbox"
    adapter._temp_dir.mkdir()
    temp_dir = adapter._temp_dir

    adapter.stop()
    adapter.stop()

    client.close.assert_awaited_once_with(force=True)
    assert not temp_dir.exists()
    assert adapter._client is None
    assert adapter._temp_dir is None
    assert not adapter._started
