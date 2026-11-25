"""CLI process handling must not use POSIX-only calls on Windows."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from private_gpt.cli.commands import run as run_cmd
from private_gpt.cli.commands import serve as serve_cmd


def _running_proc() -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 4242
    proc.poll.return_value = None
    return proc


def test_stop_server_subprocess_on_windows_terminates_without_killpg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_cmd, "_IS_WINDOWS", True)
    # os.killpg does not exist on Windows.
    killpg = MagicMock(side_effect=AssertionError("killpg on Windows"))
    monkeypatch.setattr(run_cmd.os, "killpg", killpg, raising=False)
    proc = _running_proc()

    run_cmd._stop_server_subprocess(proc)

    proc.terminate.assert_called_once()
    proc.wait.assert_called()


def test_stop_server_subprocess_on_windows_kills_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_cmd, "_IS_WINDOWS", True)
    killpg = MagicMock(side_effect=AssertionError("killpg on Windows"))
    monkeypatch.setattr(run_cmd.os, "killpg", killpg, raising=False)
    proc = _running_proc()
    proc.wait.side_effect = [subprocess.TimeoutExpired("private-gpt", 1), 0]

    run_cmd._stop_server_subprocess(proc)

    proc.kill.assert_called_once()


def test_pid_alive_on_windows_never_signals_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On Windows os.kill(pid, 0) is TerminateProcess(pid, 0): a liveness probe
    # through it would kill the server it is checking.
    monkeypatch.setattr(serve_cmd, "_IS_WINDOWS", True)
    kill = MagicMock()
    monkeypatch.setattr(serve_cmd.os, "kill", kill)
    monkeypatch.setattr(serve_cmd, "_windows_pid_alive", lambda pid: True)

    assert serve_cmd._pid_alive(1234) is True
    kill.assert_not_called()


def test_pid_alive_on_posix_detects_stale_pid() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()

    assert serve_cmd._pid_alive(proc.pid) is False
    assert serve_cmd._pid_alive(os.getpid()) is True


def test_run_attach_on_windows_does_not_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_cmd, "_IS_WINDOWS", True)
    execvp = MagicMock()
    monkeypatch.setattr(run_cmd.os, "execvp", execvp)
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(run_cmd.subprocess, "run", run)

    with pytest.raises(SystemExit) as exc:
        run_cmd._run_attached(str(tmp_path / "claude"), ["claude", "--x"])

    assert exc.value.code == 0
    execvp.assert_not_called()
    run.assert_called_once_with(["claude", "--x"], check=False)


def test_start_server_on_windows_runs_python_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The private-gpt.exe console-script launcher spawns a second Python
    # process; terminating the launcher would leave that server running.
    monkeypatch.setattr(run_cmd, "_IS_WINDOWS", True)
    monkeypatch.setattr(run_cmd.shutil, "which", lambda _: "C:/bin/private-gpt.exe")
    monkeypatch.setattr(run_cmd, "_server_log_path", lambda: tmp_path / "server.log")
    popen = MagicMock()
    monkeypatch.setattr(run_cmd.subprocess, "Popen", popen)

    run_cmd._start_server_subprocess()

    cmd = popen.call_args.args[0]
    assert cmd[:4] == [sys.executable, "-m", "private_gpt", "serve"]


def test_windows_pid_alive_treats_access_denied_as_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = MagicMock()
    kernel32.OpenProcess.return_value = 0
    monkeypatch.setattr(
        serve_cmd, "_windows_kernel32", lambda: (kernel32, lambda: 5), raising=False
    )

    assert serve_cmd._windows_pid_alive(1234) is True


def test_windows_pid_alive_treats_missing_process_as_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = MagicMock()
    kernel32.OpenProcess.return_value = 0
    monkeypatch.setattr(
        serve_cmd, "_windows_kernel32", lambda: (kernel32, lambda: 87), raising=False
    )

    assert serve_cmd._windows_pid_alive(1234) is False
