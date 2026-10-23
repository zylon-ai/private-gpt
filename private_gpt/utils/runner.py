from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version() -> str:
    try:
        return version("private-gpt")
    except PackageNotFoundError:
        version_file = Path(__file__).resolve().parents[2] / "version.txt"
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ImportError("Version metadata not found") from exc
