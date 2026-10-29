import signal
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from private_gpt.cli.commands import run
from private_gpt.cli.main import app

_ATTACH = ["my-app", "--no-server", "--attach", "--", "-p", "explain this file"]


def _invoke_run(monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(run.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(run, "_pick_model", lambda model: "default")
    monkeypatch.setattr(run, "inject_app_env", lambda model=None: None)
    return CliRunner().invoke(app, ["run", *args])


def test_attach_on_windows_waits_for_the_app_and_returns_its_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    execvp = MagicMock()
    monkeypatch.setattr(run.os, "execvp", execvp)
    app_run = MagicMock(return_value=subprocess.CompletedProcess([], 7))
    monkeypatch.setattr(run.subprocess, "run", app_run)

    result = _invoke_run(monkeypatch, *_ATTACH)

    assert result.exit_code == 7
    app_run.assert_called_once_with(
        ["/bin/my-app", "-p", "explain this file"], check=False
    )
    execvp.assert_not_called()


def test_attach_on_posix_still_replaces_the_cli_with_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    execvp = MagicMock()
    monkeypatch.setattr(run.os, "execvp", execvp)
    app_run = MagicMock()
    monkeypatch.setattr(run.subprocess, "run", app_run)

    _invoke_run(monkeypatch, *_ATTACH)

    execvp.assert_called_once_with(
        "/bin/my-app", ["/bin/my-app", "-p", "explain this file"]
    )
    app_run.assert_not_called()


def test_run_on_windows_stops_the_server_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(run.os, "killpg", raising=False)  # absent on Windows
    server = MagicMock()
    server.poll.return_value = None
    monkeypatch.setattr(run, "_is_server_up", lambda base_url: False)
    monkeypatch.setattr(run, "_start_server_subprocess", lambda: server)
    monkeypatch.setattr(run, "_wait_for_server", lambda base_url: True)
    monkeypatch.setattr(
        run.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 7)),
    )

    result = _invoke_run(monkeypatch, "my-app")

    assert result.exit_code == 7
    server.terminate.assert_called_once_with()
    server.wait.assert_called_once_with()


def test_stop_server_on_posix_still_signals_its_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    killpg = MagicMock()
    monkeypatch.setattr(run.os, "killpg", killpg, raising=False)
    server = MagicMock(pid=4321)
    server.poll.return_value = None

    run._stop_server_subprocess(server)

    killpg.assert_called_once_with(4321, signal.SIGTERM)
    server.wait.assert_called_once_with(timeout=run._SERVER_SHUTDOWN_TIMEOUT)
    server.terminate.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handling")
def test_stop_server_ends_a_real_process_on_windows() -> None:
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run._stop_server_subprocess(server)

        assert server.poll() is not None
    finally:
        server.kill()
