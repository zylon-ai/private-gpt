#!/usr/bin/env python3
"""Keep Claude-related specs in sync: OpenAPI spec URL and anthropic SDK version."""

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent

ANTHROPIC_STATS_URL = (
    "https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml"
)
ANTHROPIC_PYPI_URL = "https://pypi.org/pypi/anthropic/json"

OPENAPI_TEST_FILE = ROOT / "tests/models/anthropic/test_openapi_schema.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"


def fetch_openapi_spec_url() -> str:
    with urllib.request.urlopen(ANTHROPIC_STATS_URL, timeout=15) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("openapi_spec_url:"):
                return line.split(":", 1)[1].strip()
    raise RuntimeError(f"openapi_spec_url not found in {ANTHROPIC_STATS_URL}")


def fetch_latest_anthropic_version() -> str:
    with urllib.request.urlopen(ANTHROPIC_PYPI_URL, timeout=15) as resp:
        return json.loads(resp.read().decode())["info"]["version"]


def current_openapi_spec_url() -> str:
    source = OPENAPI_TEST_FILE.read_text(encoding="utf-8")
    match = re.search(r'^OPENAPI_SPEC_URL\s*=\s*"([^"]*)"', source, re.MULTILINE)
    return match.group(1) if match else "<not found>"


def current_anthropic_version() -> str:
    source = PYPROJECT_FILE.read_text(encoding="utf-8")
    match = re.search(r'"anthropic>=([\d.]+)"', source)
    return match.group(1) if match else "<not found>"


def update_openapi_spec_url(new_url: str) -> None:
    source = OPENAPI_TEST_FILE.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'^(OPENAPI_SPEC_URL\s*=\s*")[^"]*(")',
        rf"\g<1>{new_url}\g<2>",
        source,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError(f"OPENAPI_SPEC_URL assignment not found in {OPENAPI_TEST_FILE}")
    OPENAPI_TEST_FILE.write_text(updated, encoding="utf-8")


def update_anthropic_version(new_version: str) -> None:
    source = PYPROJECT_FILE.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'"anthropic>=[\d.]+"',
        f'"anthropic>={new_version}"',
        source,
    )
    if count == 0:
        raise RuntimeError(f'anthropic dependency not found in {PYPROJECT_FILE}')
    PYPROJECT_FILE.write_text(updated, encoding="utf-8")


def sync_lock_file() -> None:
    print("\nRunning uv lock ...")
    subprocess.run(["uv", "lock"], check=True, cwd=ROOT)


def main() -> None:
    changed = False

    # --- OpenAPI spec URL ---
    print(f"Fetching OpenAPI spec URL from {ANTHROPIC_STATS_URL} ...")
    latest_url = fetch_openapi_spec_url()
    pinned_url = current_openapi_spec_url()

    if pinned_url == latest_url:
        print("  OpenAPI spec URL already up to date.")
    else:
        rel = OPENAPI_TEST_FILE.relative_to(ROOT)
        print(f"  Updating {rel}")
        print(f"    {pinned_url}")
        print(f" -> {latest_url}")
        update_openapi_spec_url(latest_url)
        changed = True

    # --- anthropic SDK version ---
    print(f"\nFetching latest anthropic version from {ANTHROPIC_PYPI_URL} ...")
    latest_version = fetch_latest_anthropic_version()
    pinned_version = current_anthropic_version()

    if pinned_version == latest_version:
        print(f"  anthropic already at {latest_version}.")
    else:
        print(f"  Updating {PYPROJECT_FILE.name}")
        print(f"    anthropic>={pinned_version}")
        print(f" -> anthropic>={latest_version}")
        update_anthropic_version(latest_version)
        changed = True

    if not changed:
        print("\nAll specs up to date.")
        sys.exit(0)

    sync_lock_file()
    print("Done.")


if __name__ == "__main__":
    main()
