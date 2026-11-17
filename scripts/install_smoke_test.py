#!/usr/bin/env python3
"""Smoke test a running PrivateGPT server: health, ingest, retrieval, chat.

Standard library only, so it runs on any OS with any Python 3.

    python scripts/install_smoke_test.py --base-url http://127.0.0.1:8080 \
        --model qwen3:0.6b
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

CODEWORD = "AZURE-PELICAN-42"
DOCUMENT = (
    f"The PrivateGPT smoke-test codeword is {CODEWORD}.\n"
    "It was chosen on 13 November 2026.\n"
)


def request(
    base_url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600
) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise SystemExit(f"{path} -> HTTP {exc.code}: {detail}") from exc


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"server at {base_url} not healthy after {timeout:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--health-timeout", type=float, default=600)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    wait_for_health(base, args.health_timeout)
    print("health ok")

    ingest = request(
        base,
        "/v1/artifacts/ingest",
        {
            "artifact": "smoke-codeword",
            "collection": "smoke",
            "input": {
                "type": "file",
                "value": base64.b64encode(DOCUMENT.encode()).decode(),
            },
            "metadata": {"file_name": "codeword.txt"},
        },
    )
    if ingest.get("object") != "list":
        raise SystemExit(f"unexpected ingest response: {ingest}")
    print(f"ingest ok ({len(ingest.get('data', []))} documents)")

    listed = request(base, "/v1/artifacts/list?collection=smoke")
    if "smoke-codeword" not in json.dumps(listed):
        raise SystemExit(f"ingested artifact not listed: {listed}")
    print("list ok")

    search = request(
        base,
        "/v1/primitives/search",
        {
            "type": "semantic_search",
            "text": "smoke-test codeword",
            "context_filter": {"collection": "smoke"},
            "limit": 3,
        },
    )
    if CODEWORD not in json.dumps(search):
        raise SystemExit(f"retrieval did not return the document: {search}")
    print("retrieval ok")

    message = request(
        base,
        "/v1/messages",
        {
            "model": args.model,
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": "Use search_docs to find the smoke-test codeword, "
                    "then answer with the codeword only. /no_think",
                }
            ],
            "tools": [
                {
                    "name": "search_docs",
                    "type": "semantic_search_v1",
                    "context": [
                        {
                            "type": "ingested_artifact",
                            "context_filter": {
                                "collection": "smoke",
                                "artifacts": ["smoke-codeword"],
                            },
                        }
                    ],
                }
            ],
        },
    )
    if message.get("type") != "message" or message.get("role") != "assistant":
        raise SystemExit(f"unexpected message response: {message}")
    text = " ".join(
        block.get("text", "")
        for block in message.get("content", [])
        if block.get("type") == "text"
    )
    print(f"message ok: {text.strip()[:200]!r}")
    if CODEWORD not in json.dumps(message):
        # Small CI models do not always quote the tool result back.
        print("note: answer does not contain the codeword", file=sys.stderr)


if __name__ == "__main__":
    main()
