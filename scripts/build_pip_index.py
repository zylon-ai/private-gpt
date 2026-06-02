#!/usr/bin/env python3
"""Generate a static PEP 503-style package index from built distributions."""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path


@dataclass(frozen=True)
class Distribution:
    name: str
    normalized_name: str
    filename: str
    requires_python: str | None
    sha256: str


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_metadata(payload: str) -> tuple[str, str | None]:
    metadata = Parser().parsestr(payload)
    name = metadata["Name"]
    if not name:
        raise ValueError("Distribution metadata is missing Name")
    return name, metadata.get("Requires-Python")


def read_wheel_metadata(path: Path) -> tuple[str, str | None]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        return parse_metadata(archive.read(metadata_name).decode())


def read_sdist_metadata(path: Path) -> tuple[str, str | None]:
    with tarfile.open(path, "r:gz") as archive:
        pkg_info = next(
            member
            for member in archive.getmembers()
            if member.name == "PKG-INFO" or member.name.endswith("/PKG-INFO")
        )
        fileobj = archive.extractfile(pkg_info)
        if fileobj is None:
            raise ValueError(f"Unable to read PKG-INFO from {path}")
        return parse_metadata(fileobj.read().decode())


def read_distribution(path: Path) -> Distribution:
    if path.suffix == ".whl":
        name, requires_python = read_wheel_metadata(path)
    elif path.name.endswith(".tar.gz"):
        name, requires_python = read_sdist_metadata(path)
    else:
        raise ValueError(f"Unsupported distribution format: {path.name}")

    return Distribution(
        name=name,
        normalized_name=normalize_name(name),
        filename=path.name,
        requires_python=requires_python,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def render_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
  </head>
  <body>
    <h1>{html.escape(title)}</h1>
    {body}
  </body>
</html>
"""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_index(package_dir: Path, output_dir: Path, extra: str) -> None:
    package_dir = package_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_dir = output_dir / "simple"
    if simple_dir.exists():
        for path in sorted(simple_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    simple_dir.mkdir(exist_ok=True)

    distributions = [
        read_distribution(path)
        for path in sorted(package_dir.iterdir())
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    ]
    if not distributions:
        raise ValueError(f"No distributions found in {package_dir}")

    grouped: dict[str, list[Distribution]] = defaultdict(list)
    canonical_names: dict[str, str] = {}
    for distribution in distributions:
        grouped[distribution.normalized_name].append(distribution)
        canonical_names.setdefault(distribution.normalized_name, distribution.name)

    package_links = []
    artifact_links = []
    for normalized_name in sorted(grouped):
        canonical_name = canonical_names[normalized_name]
        project_dir = simple_dir / normalized_name
        project_dir.mkdir(parents=True, exist_ok=True)

        file_links = []
        for distribution in sorted(
            grouped[normalized_name], key=lambda item: item.filename
        ):
            filename = html.escape(distribution.filename)
            href = f"../../packages/{filename}#sha256={distribution.sha256}"
            requires_python = ""
            if distribution.requires_python:
                requires_python = f' data-requires-python="{html.escape(distribution.requires_python)}"'
            file_links.append(f'<a href="{href}"{requires_python}>{filename}</a><br>')

        write_text(
            project_dir / "index.html",
            render_page(canonical_name, "\n    ".join(file_links)),
        )
        package_links.append(
            f'<a href="{normalized_name}/">{html.escape(canonical_name)}</a><br>'
        )

    packages_dir = output_dir / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    for distribution in sorted(distributions, key=lambda item: item.filename):
        filename = html.escape(distribution.filename)
        artifact_links.append(f'<a href="{filename}">{filename}</a><br>')

    write_text(
        packages_dir / "index.html",
        render_page("PrivateGPT package files", "\n    ".join(artifact_links)),
    )

    write_text(
        simple_dir / "index.html",
        render_page("PrivateGPT package index", "\n    ".join(package_links)),
    )
    write_text(
        output_dir / "index.html",
        render_page(
            "PrivateGPT package index",
            (
                "<p>Install PrivateGPT with <code>uv tool install --find-links "
                "https://zylon-ai.github.io/private-gpt/packages/ "
                f"&quot;private-gpt[{html.escape(extra)}]&quot;</code>.</p>"
            ),
        ),
    )
    (output_dir / ".nojekyll").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra", default="core")
    args = parser.parse_args()

    build_index(Path(args.package_dir), Path(args.output_dir), args.extra)


if __name__ == "__main__":
    main()
