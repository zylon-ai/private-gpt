"""Run an end-to-end Anthropic streaming check against Zylon + ARQ.

The script starts the Zylon API and PrivateGPT ARQ worker, streams a normal
message, then forces the built-in code-execution tool. It fails on process
crashes, request timeouts, Anthropic error events, missing tool results, or
leftover resumable-chat Redis keys.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import anthropic
import httpx
import redis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "zylon-gpt",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="default")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--skip-simple", action="store_true")
    parser.add_argument("--skip-tool", action="store_true")
    return parser


class _Process:
    def __init__(self, name: str, command: list[str], cwd: Path, env: dict[str, str]):
        self.name = name
        self.lines: deque[str] = deque(maxlen=400)
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self.process.stdout is not None
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            rendered = f"[{self.name}] {line.rstrip()}"
            self.lines.append(rendered)
            print(rendered, flush=True)

    def assert_running(self) -> None:
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                f"{self.name} exited with code {code}\n" + "\n".join(self.lines)
            )

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)


def _wait_for_server(url: str, processes: list[_Process], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        for process in processes:
            process.assert_running()
        try:
            response = httpx.get(f"{url}/openapi.json", timeout=3)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready: {last_error}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stream_message(
    client: anthropic.Anthropic,
    *,
    model: str,
    prompt: str,
    timeout: float,
    tools: list[Any] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    event_types: list[str] = []
    block_types: list[str] = []
    execution_ids: list[str] = []
    started = time.monotonic()
    stream_method = cast(Any, client.messages.stream)
    with stream_method(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    ) as stream:
        for event in stream:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Anthropic stream exceeded {timeout} seconds")
            event_type = getattr(event, "type", "")
            event_types.append(event_type)
            if event_type == "error":
                raise RuntimeError(f"Anthropic stream returned error: {event}")
            if event_type == "message_start":
                message_id = getattr(getattr(event, "message", None), "id", None)
                if message_id:
                    execution_ids.append(message_id)
            if event_type == "content_block_start":
                content_block = getattr(event, "content_block", None)
                block_types.append(getattr(content_block, "type", ""))

    if not event_types or event_types[-1] != "message_stop":
        raise AssertionError(f"Incomplete event sequence: {event_types}")
    return event_types, block_types, execution_ids


def _assert_redis_cleanup(env: dict[str, str], execution_ids: list[str]) -> None:
    host = env.get("PGPT_REDIS_HOST", "localhost")
    if ":" in host:
        hostname, port = host.rsplit(":", 1)
    else:
        hostname, port = host, "6379"
    database = int(env.get("PGPT_REDIS_DATABASE", "0") or 0) + 8
    client = redis.Redis(
        host=hostname,
        port=int(port),
        db=database,
        username=env.get("PGPT_REDIS_USERNAME") or None,
        password=env.get("PGPT_REDIS_PASSWORD") or None,
        decode_responses=True,
        socket_timeout=5,
    )
    leftovers: list[str] = []
    for execution_id in execution_ids:
        leftovers.extend(
            client.scan_iter(f"private_gpt:arq:iteration:*:{execution_id}")
        )
        leftovers.extend(client.scan_iter(f"private_gpt:engine:events:{execution_id}"))
    if leftovers:
        raise AssertionError(f"Leftover resumable-chat Redis keys: {leftovers}")


def main() -> None:
    args = _parser().parse_args()
    project_dir = args.project_dir.resolve()
    base_url = args.base_url or f"http://127.0.0.1:{_free_port()}/gpt"
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    parsed_url = urlparse(base_url)
    env.update(
        {
            "PGPT_PROFILES": "local,me",
            "PGPT_SETTINGS_FOLDER": ".,../private-gpt",
            "PGPT_WORKER_APP_MODULE": "zylon_gpt",
            "PGPT_WORKER_MODE": "arq",
            "PGPT_CHAT_ENGINE_MODE": "async",
            "PGPT_CHAT_SCHEDULER_MODE": "arq",
            "PGPT_TOOLS_SCHEDULER_MODE": "celery",
            "PGPT_CONDENSE_CHAT_HISTORY": "none",
            "PYTHONUNBUFFERED": "1",
            "API_ENABLED": "false",
            "PORT": str(parsed_url.port or 18081),
        }
    )
    celery_env = {
        **env,
        "PGPT_WORKER_MODE": "worker",
        "PGPT_CELERY_QUEUES": "tools",
        "PGPT_CELERY_POOL": "solo",
    }
    processes = [
        _Process(
            "server",
            [
                "uv",
                "run",
                "uvicorn",
                "zylon_gpt.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(parsed_url.port or 18081),
                "--loop",
                "asyncio",
                "--no-access-log",
            ],
            project_dir,
            env,
        ),
        _Process(
            "arq-worker",
            ["uv", "run", "private-gpt", "worker"],
            project_dir,
            env,
        ),
        _Process(
            "celery-tools-worker",
            ["uv", "run", "private-gpt", "worker"],
            project_dir,
            celery_env,
        ),
    ]
    try:
        _wait_for_server(base_url, processes, args.startup_timeout)
        for process in processes:
            process.assert_running()
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "integration-test"),
            base_url=base_url,
            timeout=args.request_timeout,
        )
        execution_ids: list[str] = []
        if not args.skip_simple:
            event_types, _, message_ids = _stream_message(
                client,
                model=args.model,
                prompt="Reply with exactly: async streaming works",
                timeout=args.request_timeout,
            )
            execution_ids.extend(message_ids)
            print(f"Simple stream passed with {len(event_types)} events")

        if not args.skip_tool:
            tool = {
                "type": "code_execution_20250825",
                "name": "code_execution",
            }
            _, block_types, message_ids = _stream_message(
                client,
                model=args.model,
                prompt=(
                    "You must use the code_execution tool to run Python that prints "
                    "the exact text ARQ_TOOL_OK, then report that output."
                ),
                timeout=args.request_timeout,
                tools=[tool],
            )
            execution_ids.extend(message_ids)
            if "tool_use" not in block_types or "tool_result" not in block_types:
                raise AssertionError(
                    f"Tool did not execute and resume correctly: {block_types}"
                )
            print(f"Tool stream passed with blocks: {block_types}")

        for process in processes:
            process.assert_running()
        _assert_redis_cleanup(env, execution_ids)
        print("ARQ Anthropic streaming integration passed")
    finally:
        for process in reversed(processes):
            process.stop()


if __name__ == "__main__":
    main()
