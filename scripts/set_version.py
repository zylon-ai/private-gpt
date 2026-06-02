#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_TXT = REPO_ROOT / "version.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the local project version across release-managed files."
    )
    parser.add_argument(
        "version", help="Version to set, for example 1.0.0 or 1.0.0-rc1"
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip refreshing uv.lock with `uv lock`.",
    )
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def update_version_txt(version: str) -> None:
    write_text(VERSION_TXT, f"{version}\n")


def update_pyproject(version: str) -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version = "[^"]+"$',
        f'version = "{version}"',
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Failed to update project version in pyproject.toml")
    write_text(PYPROJECT, updated)


def refresh_uv_lock() -> None:
    result = subprocess.run(
        ["uv", "lock"],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("`uv lock` failed")


def main() -> int:
    args = parse_args()

    update_version_txt(args.version)
    update_pyproject(args.version)

    if not args.no_lock:
        refresh_uv_lock()

    print(f"Updated local version to {args.version}")
    if args.no_lock:
        print("Skipped uv.lock refresh")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
