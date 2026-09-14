from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from private_gpt.components.code_execution.bash_executor import LocalBashExecutor
from private_gpt.components.sandbox.local import (
    BashExecutorSandbox,
    LocalSandboxSession,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("force", [False, True])
async def test_bash_executor_sandbox_close_accepts_force(force: bool) -> None:
    sandbox = BashExecutorSandbox([], MagicMock(spec=LocalBashExecutor))

    await sandbox.close(force=force)


@pytest.mark.parametrize("force", [False, True])
async def test_local_sandbox_close_removes_workspace(
    tmp_path: Path, force: bool
) -> None:
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "result.txt").write_text("result")
    sandbox = LocalSandboxSession([], MagicMock(spec=LocalBashExecutor), workdir)

    await sandbox.close(force=force)

    assert not workdir.exists()
