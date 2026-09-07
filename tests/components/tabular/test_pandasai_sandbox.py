from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.sandbox.base import SandboxExecutionResult, SandboxSession
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


def test_start_generates_valid_setup_code() -> None:
    client = AsyncMock(spec=SandboxSession)
    client.run_code.return_value = SandboxExecutionResult(success=True)
    adapter = PandasAISandboxAdapter(client=client)

    try:
        adapter.start()
        setup_code = client.run_code.call_args.args[0]
        compile(setup_code, "<sandbox setup>", "exec")
        assert adapter._started
    finally:
        adapter.stop()


@pytest.mark.parametrize("raises", [False, True], ids=["failed-result", "exception"])
def test_start_failure_propagates_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    client = AsyncMock(spec=SandboxSession)
    if raises:
        client.run_code.side_effect = RuntimeError("setup failed")
    else:
        client.run_code.return_value = SandboxExecutionResult(
            success=False, stderr="setup failed"
        )
    adapter = PandasAISandboxAdapter(client=client)
    temp_dir = tmp_path / "sandbox"
    temp_dir.mkdir()
    monkeypatch.setattr(
        "private_gpt.components.tabular.pandasai_sandbox.tempfile.mkdtemp",
        lambda **kwargs: str(temp_dir),
    )

    try:
        with pytest.raises(RuntimeError, match="setup failed"):
            adapter.start()
        client.close.assert_awaited_once_with(force=True)
        assert not temp_dir.exists()
    finally:
        adapter.stop()

    client.close.assert_awaited_once_with(force=True)
    assert not temp_dir.exists()
    assert adapter._client is None
    assert adapter._temp_dir is None
    assert not adapter._started


def test_stop_cleans_up_partially_initialized_session(tmp_path: Path) -> None:
    client = AsyncMock(spec=SandboxSession)
    adapter = PandasAISandboxAdapter(client=client)
    temp_dir = tmp_path / "sandbox"
    temp_dir.mkdir()
    adapter._temp_dir = temp_dir

    adapter.stop()

    client.close.assert_awaited_once_with(force=True)
    assert not temp_dir.exists()
    assert adapter._client is None
    assert adapter._temp_dir is None
    assert not adapter._started
