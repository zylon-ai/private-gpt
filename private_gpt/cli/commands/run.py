import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import typer

from private_gpt.settings.settings import settings

_APP_BINARIES: dict[str, str] = {
    "claude-code": "claude",
    "openclaw": "openclaw",
    "opencode": "opencode",
}

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_TIMEOUT = 60.0
_SERVER_SHUTDOWN_TIMEOUT = 10.0


def _base_url() -> str:
    s = settings().server
    url = f"http://localhost:{s.port}"
    if s.root_path:
        url += f"/{s.root_path.strip('/')}".rstrip("/")
    return url


def _is_server_up(base_url: str) -> bool:
    try:
        import httpx

        r = httpx.get(f"{base_url}/health", timeout=2.0)
        return r.is_success
    except Exception:
        return False


def _wait_for_server(base_url: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_server_up(base_url):
            return True
        time.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _start_server_subprocess() -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        [sys.executable, "-m", "private_gpt", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_server_subprocess(server_proc: "subprocess.Popen[bytes]") -> None:
    if server_proc.poll() is not None:
        return

    try:
        os.killpg(server_proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        server_proc.wait(timeout=_SERVER_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(server_proc.pid, signal.SIGKILL)
        server_proc.wait()


def _inject_opencode_env(model: str | None = None) -> None:
    import json

    from private_gpt.settings.settings import LLMModelConfig

    s = settings()
    base_url = _base_url()
    api_key = s.server.auth.secret if s.server.auth.enabled else "no-auth"

    llm_models = {
        alias: {
            "name": f"PrivateGPT - {alias}",
            "limit": {
                "context": m.context_window,
                "output": m.sampling_params.max_new_tokens,
            },
        }
        for m in s.models
        for alias in [m.name, m.alias]
        if isinstance(m, LLMModelConfig)
        if alias
    }
    default_model = model or s.llm.default_model or next(iter(s.models)).name
    default_model = f"PrivateGPT - {default_model}"

    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "anthropic": {
                "name": "PrivateGPT",
                "options": {
                    "baseURL": f"{base_url}/v1",
                    "apiKey": api_key,
                },
                "models": llm_models,
            }
        },
        "model": default_model,
        "small_model": default_model,
    }

    config_dir = Path.home() / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "private-gpt.json"
    config_path.write_text(json.dumps(config, indent=2))
    os.environ["OPENCODE_CONFIG"] = str(config_path)
    os.environ["ANTHROPIC_BASE_URL"] = f"{base_url}/v1"
    os.environ["ANTHROPIC_API_KEY"] = api_key


def inject_app_env(model: str | None = None) -> None:
    s = settings().server
    base_url = _base_url()
    api_key = s.auth.secret if s.auth.enabled else "no-auth"
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if model:
        os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model


def run_command(
    ctx: typer.Context,
    app_name: str = typer.Argument(
        ..., metavar="app", help="claude-code | openclaw | custom"
    ),
    attach: bool
    | None = typer.Option(
        None,
        "--attach/--no-attach",
        help="Attach stdin/stdout (default when TTY detected)",
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Run in background, return run ID"
    ),
    session: str
    | None = typer.Option(None, "--session", help="Resume a previous app session"),
    model: str | None = typer.Option("default", "--model", help="Model name to use"),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", help="Skip confirmations (CI mode)"
    ),
    no_server: bool = typer.Option(
        False, "--no-server", help="Skip server detection and auto-start entirely"
    ),
) -> None:
    """Launch a connected app (claude-code, openclaw, custom).

    Pass extra arguments to the app after --:

        private-gpt run claude-code -- --resume abc123
    """
    extra_args: list[str] = ctx.args

    binary = _APP_BINARIES.get(app_name, app_name)
    resolved = shutil.which(binary)
    if resolved is None:
        typer.echo(f"Binary not found in PATH: {binary!r}", err=True)
        raise SystemExit(1)

    base_url = _base_url()
    server_proc: subprocess.Popen[bytes] | None = None

    if not no_server and not _is_server_up(base_url):
        typer.echo("Server not reachable, starting automatically...")
        server_proc = _start_server_subprocess()
        if not _wait_for_server(base_url):
            server_proc.terminate()
            typer.echo(
                f"Server did not become ready within {_HEALTH_TIMEOUT}s", err=True
            )
            raise SystemExit(1)
        typer.echo("Server is ready.")

    match app_name:
        case "opencode":
            _inject_opencode_env(model)
        case _:
            inject_app_env(model)

    cmd: list[str] = [resolved]
    if session:
        cmd += ["--resume", session]
    if auto_approve:
        cmd.append("--yes")
    cmd += extra_args

    if detach:
        run_id = uuid.uuid4().hex[:8]
        log_dir = Path.home() / ".private-gpt" / "runs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"
        with log_path.open("wb") as log_fh:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        typer.echo(run_id)
        return

    is_tty = sys.stdin.isatty()
    use_attach = attach if attach is not None else is_tty

    if server_proc is not None:
        try:
            result = subprocess.run(cmd, check=False)
            raise SystemExit(result.returncode)
        finally:
            _stop_server_subprocess(server_proc)

    if use_attach:
        os.execvp(resolved, cmd)
    else:
        result = subprocess.run(cmd, check=False)
        raise SystemExit(result.returncode)
