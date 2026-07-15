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


def _server_log_path() -> Path:
    log_dir = Path.home() / ".private-gpt"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "server.log"


def _start_server_subprocess() -> "subprocess.Popen[bytes]":
    s = settings().server
    private_gpt_bin = shutil.which("private-gpt")
    base_cmd = (
        [private_gpt_bin, "serve"]
        if private_gpt_bin
        else [sys.executable, "-m", "private_gpt", "serve"]
    )
    cmd = [*base_cmd, "--host", s.host, "--port", str(s.port)]
    log_path = _server_log_path()
    typer.echo(f"Server logs: {log_path}")
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    return proc


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


_OPENCODE_FALLBACK: dict[str, Any] = {
    "id": "default",
    "context": 8096,
    "output": 1024,
}


def _fetch_llm_models(base_url: str, api_key: str) -> list[dict[str, Any]]:
    import httpx

    try:
        r = httpx.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        r.raise_for_status()
        return [m for m in r.json().get("data", []) if m.get("embed_dim") is None]
    except Exception:
        return []


_MODEL_PAGE_SIZE = 20


def _render_model_page(
    models: list[dict[str, Any]], page: int, total_pages: int
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Available Models  (page {page + 1}/{total_pages})")
    table.add_column("#", style="bold cyan", justify="right", width=4)
    table.add_column("Model ID", style="white")
    table.add_column("Context", justify="right", style="green")
    table.add_column("Max output", justify="right", style="yellow")

    offset = page * _MODEL_PAGE_SIZE
    for i, m in enumerate(models, offset + 1):
        ctx = str(m["max_input_tokens"]) if m.get("max_input_tokens") else "-"
        out = str(m["max_tokens"]) if m.get("max_tokens") else "-"
        table.add_row(str(i), m["id"], ctx, out)

    console.print(table)


def _pick_model(model: str | None) -> str:
    s = settings()
    base_url = _base_url()
    api_key = s.server.auth.secret if s.server.auth.enabled else "no-auth"

    models = _fetch_llm_models(base_url, api_key)

    if model is not None:
        return model

    if not models:
        default_model: str = _OPENCODE_FALLBACK["id"]
        return default_model

    if len(models) == 1:
        default_id: str = models[0]["id"]
        return default_id

    total_pages = (len(models) + _MODEL_PAGE_SIZE - 1) // _MODEL_PAGE_SIZE
    page = 0

    while True:
        start = page * _MODEL_PAGE_SIZE
        end = start + _MODEL_PAGE_SIZE
        page_models = models[start:end]

        _render_model_page(page_models, page, total_pages)

        nav = []
        if page > 0:
            nav.append("[p] prev")
        if page < total_pages - 1:
            nav.append("[n] next")
        nav.append("[1-N] pick")

        raw = typer.prompt(f"  {' · '.join(nav)}", default="1").strip().lower()

        if raw == "n" and page < total_pages - 1:
            page += 1
            continue
        if raw == "p" and page > 0:
            page -= 1
            continue

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                model_id: str = models[idx]["id"]
                return model_id
            typer.echo(f"  Pick a number between 1 and {len(models)}.", err=True)
        except ValueError:
            if any(m["id"] == raw for m in models):
                model_raw: str = raw
                return model_raw
            typer.echo(f"  Unknown model {raw!r}.", err=True)


def _inject_opencode_env(model: str | None = None) -> None:
    import json

    s = settings()
    base_url = _base_url()
    api_key = s.server.auth.secret if s.server.auth.enabled else "no-auth"

    llm_models: dict[str, Any] = {
        m["id"]: {
            "name": f"PrivateGPT - {m['id']}",
            "limit": {
                "context": m.get("max_input_tokens") or _OPENCODE_FALLBACK["context"],
                "output": m.get("max_tokens") or _OPENCODE_FALLBACK["output"],
            },
        }
        for m in _fetch_llm_models(base_url, api_key)
    }

    if not llm_models:
        fid = _OPENCODE_FALLBACK["id"]
        llm_models[fid] = {
            "name": f"PrivateGPT - {fid}",
            "limit": {
                "context": _OPENCODE_FALLBACK["context"],
                "output": _OPENCODE_FALLBACK["output"],
            },
        }

    first_model_id = next(iter(llm_models))
    default_model = model or s.llm.default_model or first_model_id
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
    attach: bool | None = typer.Option(
        None,
        "--attach/--no-attach",
        help="Attach stdin/stdout (default when TTY detected)",
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Run in background, return run ID"
    ),
    session: str | None = typer.Option(
        None, "--session", help="Resume a previous app session"
    ),
    model: str | None = typer.Option(None, "--model", help="Model name to use"),
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
            inject_app_env(_pick_model(model))

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
