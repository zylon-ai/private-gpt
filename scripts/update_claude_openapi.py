#!/usr/bin/env python3
"""Update OPENAPI_SPEC_URL in test_openapi_schema.py from the upstream .stats.yml."""

import re
import sys
import urllib.request
from pathlib import Path

STATS_URL = "https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml"
TEST_FILE = (
    Path(__file__).parent.parent / "tests/models/anthropic/test_openapi_schema.py"
)


def fetch_openapi_spec_url() -> str:
    with urllib.request.urlopen(STATS_URL, timeout=15) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("openapi_spec_url:"):
                spec: str = line.split(":", 1)[1].strip()
                return spec
    raise RuntimeError("openapi_spec_url not found in .stats.yml")


def update_test_file(new_url: str) -> None:
    source = TEST_FILE.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'^(OPENAPI_SPEC_URL\s*=\s*")[^"]*(")',
        rf"\g<1>{new_url}\g<2>",
        source,
        flags=re.MULTILINE,
    )
    if count == 0:
        raise RuntimeError(f"OPENAPI_SPEC_URL assignment not found in {TEST_FILE}")
    TEST_FILE.write_text(updated, encoding="utf-8")


def main() -> None:
    print(f"Fetching {STATS_URL} ...")
    new_url = fetch_openapi_spec_url()
    print(f"openapi_spec_url: {new_url}")

    source = TEST_FILE.read_text(encoding="utf-8")
    match = re.search(r'^OPENAPI_SPEC_URL\s*=\s*"([^"]*)"', source, re.MULTILINE)
    current_url = match.group(1) if match else "<not found>"

    if current_url == new_url:
        print("Already up to date.")
        sys.exit(0)

    print(f"Updating {TEST_FILE.relative_to(Path(__file__).parent.parent)} ...")
    print(f"  old: {current_url}")
    print(f"  new: {new_url}")
    update_test_file(new_url)
    print("Done.")


if __name__ == "__main__":
    main()
