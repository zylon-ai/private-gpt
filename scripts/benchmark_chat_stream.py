"""Measure the latency PrivateGPT adds on top of a streaming LLM backend.

Starts a fake OpenAI-compatible backend that streams a fixed number of tokens
with a fixed time to first token and inter-token delay, starts PrivateGPT
against it (``PGPT_PROFILES=benchmark``), then fires N concurrent
``/v1/messages`` requests and reports how much latency PrivateGPT added over
the backend's own timing.

    uv run python scripts/benchmark_chat_stream.py --concurrency 50
    uv run python scripts/benchmark_chat_stream.py --concurrency 50 --no-stream

Exits non-zero when ``--max-added-ttft-ms`` / ``--max-added-total-ms`` are
exceeded, so it can gate CI.
"""

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PGPT_HOME = Path(os.environ.get("PGPT_HOME", "~/.local/share/private-gpt")).expanduser()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


BACKEND_ARRIVALS: list[float] = []
# Measured backend stream durations, so timer drift is not counted as overhead.
BACKEND_DURATIONS: list[float] = []


def fake_backend(tokens: int, ttft_s: float, itl_s: float) -> FastAPI:
    """An OpenAI Chat Completions endpoint that streams ``tokens`` words."""
    app = FastAPI()

    def chunk(delta: dict[str, object], finish: str | None = None) -> bytes:
        body = {
            "id": "chatcmpl-bench",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "bench",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> object:
        BACKEND_ARRIVALS.append(time.perf_counter())
        payload = await request.json()

        async def stream() -> AsyncIterator[bytes]:
            started = time.perf_counter()
            await asyncio.sleep(ttft_s)
            yield chunk({"role": "assistant", "content": ""})
            for index in range(tokens):
                yield chunk({"content": f"word{index} "})
                await asyncio.sleep(itl_s)
            yield chunk({}, "stop")
            usage = {
                "id": "chatcmpl-bench",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "bench",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": tokens,
                    "total_tokens": tokens + 10,
                },
            }
            yield f"data: {json.dumps(usage)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            BACKEND_DURATIONS.append(time.perf_counter() - started)

        if payload.get("stream"):
            return StreamingResponse(stream(), media_type="text/event-stream")
        await asyncio.sleep(ttft_s + itl_s * tokens)
        return {
            "id": "chatcmpl-bench",
            "object": "chat.completion",
            "created": 0,
            "model": "bench",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " ".join(f"word{i}" for i in range(tokens)),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": tokens},
        }

    return app


SETTINGS = """\
server:
  env_name: benchmark
  auth:
    enabled: false
observability:
  mode: none
llm:
  default_model: bench
  auto_discover_models: false
embedding:
  auto_discover_models: false
models:
  - name: bench
    type: llm
    mode: openai
    context_window: 32768
    prompt_style: chat
    tokenizer_mode: huggingface
    tokenizer: {tokenizer}
    support_tools: false
  - name: mock
    type: embedding
    mode: mock
    context_window: 384
openai:
  api_base: {api_base}
  api_key: bench
data:
  local_data_folder: {data}
qdrant:
  path: {data}/qdrant
  url: null
vectorstore:
  database: qdrant
  embed_dim: 384
"""


@dataclass
class Sample:
    start: float
    ttft: float
    total: float
    events: int


async def _one_request(
    client: httpx.AsyncClient, url: str, stream: bool, prompt: str
) -> Sample:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 1024,
    }
    start = time.perf_counter()
    ttft = 0.0
    events = 0
    if stream:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events += 1
                    if not ttft and '"text_delta"' in line:
                        ttft = time.perf_counter() - start
    else:
        response = await client.post(url, json=body)
        response.raise_for_status()
        ttft = time.perf_counter() - start
        events = 1
    return Sample(
        start=start, ttft=ttft, total=time.perf_counter() - start, events=events
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile / 100 * (len(ordered) - 1)))
    return ordered[index]


async def _wait_ready(url: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"{url} did not become ready")


async def run(args: argparse.Namespace) -> int:
    backend_port = _free_port()
    app_port = args.app_port or _free_port()
    backend = uvicorn.Server(
        uvicorn.Config(
            fake_backend(args.tokens, args.ttft_ms / 1000, args.itl_ms / 1000),
            host="127.0.0.1",
            port=backend_port,
            log_level="warning",
        )
    )
    backend_task = asyncio.create_task(backend.serve())

    server: subprocess.Popen[bytes] | None = None
    with (
        tempfile.TemporaryDirectory(prefix="pgpt-bench-") as scratch,
        open(Path(scratch) / "server.log", "wb") as log,
    ):
        if not args.app_port:
            (Path(scratch) / "settings-benchmark.yaml").write_text(
                SETTINGS.format(
                    api_base=f"http://127.0.0.1:{backend_port}/v1",
                    tokenizer=args.tokenizer,
                    data=f"{scratch}/data",
                )
            )
            # PrivateGPT reads tokenizers from $PGPT_HOME/models/cache; reuse the
            # Hugging Face hub cache instead of downloading again.
            (Path(scratch) / "models").mkdir()
            (Path(scratch) / "models" / "cache").symlink_to(
                Path(args.hf_cache).expanduser()
            )
            nltk_cache = PGPT_HOME / "models" / "nltk_cache"
            if nltk_cache.is_dir():
                (Path(scratch) / "models" / "nltk_cache").symlink_to(nltk_cache)
            env = {
                **os.environ,
                "PGPT_HOME": scratch,
                "HF_HUB_OFFLINE": "1",
                "PGPT_PROFILES": "benchmark",
                "PGPT_SETTINGS_FOLDER": f"{PROJECT_ROOT},{scratch}",
                "PORT": str(app_port),
            }
            server = subprocess.Popen(
                [
                    *args.server_prefix.split(),
                    sys.executable,
                    *(args.server_script.split() or ["-m", "uvicorn"]),
                    "private_gpt.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(app_port),
                    "--log-level",
                    "warning",
                    "--loop",
                    args.loop,
                    *args.uvicorn_args,
                ],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        try:
            base = f"http://127.0.0.1:{app_port}"
            try:
                await _wait_ready(f"{base}/health")
            except TimeoutError:
                log.flush()
                sys.stderr.write((Path(scratch) / "server.log").read_text()[-4000:])
                raise
            url = f"{base}/v1/messages"

            limits = httpx.Limits(max_connections=args.concurrency * 2)
            async with httpx.AsyncClient(timeout=300, limits=limits) as client:
                # Warm up imports, model registry and connection pools.
                await _one_request(client, url, not args.no_stream, "warm up")
                if server is not None and args.signal_after_warmup:
                    server.send_signal(args.signal_after_warmup)
                samples: list[Sample] = []
                BACKEND_ARRIVALS.clear()
                BACKEND_DURATIONS.clear()
                for _ in range(args.rounds):
                    samples += await asyncio.gather(
                        *(
                            _one_request(
                                client, url, not args.no_stream, f"question {i}"
                            )
                            for i in range(args.concurrency)
                        )
                    )
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            backend.should_exit = True
            await backend_task

    # Time from sending the request to the backend receiving it (pre-LLM work).
    starts = sorted(s.start for s in samples)
    arrivals = sorted(BACKEND_ARRIVALS)[: len(starts)]
    pre_llm = [(a - s) * 1000 for s, a in zip(starts, arrivals, strict=False)]
    added_ttft = [(s.ttft - args.ttft_ms / 1000) * 1000 for s in samples]
    nominal = args.ttft_ms / 1000 + args.itl_ms / 1000 * args.tokens
    backend_total = (
        statistics.median(BACKEND_DURATIONS) if BACKEND_DURATIONS else nominal
    )
    added_total = [(s.total - backend_total) * 1000 for s in samples]
    report = {
        "concurrency": args.concurrency,
        "stream": not args.no_stream,
        "tokens": args.tokens,
        "requests": len(samples),
        "added_ttft_ms_p50": round(statistics.median(added_ttft), 1),
        "added_ttft_ms_p95": round(_percentile(added_ttft, 95), 1),
        "added_total_ms_p50": round(statistics.median(added_total), 1),
        "added_total_ms_p95": round(_percentile(added_total, 95), 1),
        "backend_ms_p50": round(backend_total * 1000, 1),
        "pre_llm_ms_p50": round(statistics.median(pre_llm), 1) if pre_llm else None,
        "backend_calls_per_request": round(len(BACKEND_ARRIVALS) / len(samples), 2),
        "events_per_request": round(statistics.mean(s.events for s in samples), 1),
    }
    print(json.dumps(report))
    failed = (
        args.max_added_ttft_ms is not None
        and report["added_ttft_ms_p95"] > args.max_added_ttft_ms
    ) or (
        args.max_added_total_ms is not None
        and report["added_total_ms_p95"] > args.max_added_total_ms
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--ttft-ms", type=float, default=100.0)
    parser.add_argument("--itl-ms", type=float, default=10.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default="nvidia/Qwen3.8-Flash-Next-NVFP4",
        help="Hugging Face tokenizer for the chat template and token counts.",
    )
    parser.add_argument("--hf-cache", default="~/.cache/huggingface/hub")
    parser.add_argument(
        "--server-prefix",
        default="",
        help="Command to wrap the server with, e.g. a profiler.",
    )
    parser.add_argument(
        "--signal-after-warmup",
        type=int,
        default=0,
        help="Signal sent to the server after warm-up (e.g. to start a profiler).",
    )
    parser.add_argument(
        "--server-script",
        default="",
        help="Python script that serves the app instead of `-m uvicorn`.",
    )
    parser.add_argument(
        "--app-port",
        type=int,
        default=0,
        help="Benchmark an already running server instead of starting one.",
    )
    parser.add_argument(
        "--loop", default="auto", help="uvicorn event loop (auto, asyncio or uvloop)."
    )
    parser.add_argument("--max-added-ttft-ms", type=float)
    parser.add_argument("--max-added-total-ms", type=float)
    parser.add_argument(
        "uvicorn_args", nargs="*", help="Extra uvicorn arguments, after --."
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
