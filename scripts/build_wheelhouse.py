#!/usr/bin/env python3
"""Build an offline, lock-pinned install bundle for the current platform.

The bundle holds every wheel ``private-gpt[<extra>]`` needs, at the exact
versions in ``uv.lock``. Run it with Python 3.11 on each target platform (wheels are
platform-specific), with pip and uv available:

    uv run --no-project --python 3.11 --with pip \
        python scripts/build_wheelhouse.py --project-wheel dist/private_gpt-X.whl

Output: ``<out-dir>/private-gpt-<version>-<extra>-<platform>.{tar.gz,zip}``
containing ``wheelhouse/``, a hashed ``requirements.txt`` and ``INSTALL.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_HASH_LINE = re.compile(r"^\s+--hash=sha256:[0-9a-f]{64}(\s*\\)?$")
_REQ_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def platform_tag() -> str:
    system = {"darwin": "macos", "win32": "windows"}.get(sys.platform, sys.platform)
    return f"{system}-{platform.machine().lower()}"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def export_requirements(extra: str, dest: Path) -> None:
    """Export the locked, hashed dependency set of ``private-gpt[extra]``."""
    run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--extra",
            extra,
            "--format",
            "requirements.txt",
            "--no-header",
            "--no-annotate",
            "-o",
            str(dest),
        ],
        cwd=REPO_ROOT,
    )


def wheel_name_version(path: Path) -> tuple[str, str]:
    name, version = path.name.split("-")[:2]
    return normalize(name), version


def rewrite_hashes(requirements: str, built: dict[str, Path]) -> str:
    """Point sdist-only packages at the wheels built from them.

    pip verifies the hashed sdists while downloading; the bundle then ships
    wheels instead so installing needs no build backend and no network.
    """
    out: list[str] = []
    skipping = False
    for line in requirements.splitlines():
        if skipping and _HASH_LINE.match(line):
            continue
        skipping = False
        match = _REQ_LINE.match(line)
        if match and normalize(match.group(1)) in built:
            wheel = built[normalize(match.group(1))]
            head = line.split("\\")[0].rstrip()
            out.append(f"{head} \\")
            out.append(f"    --hash=sha256:{sha256(wheel)}")
            skipping = True
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def build(project_wheel: Path, extra: str, out_dir: Path) -> Path:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("build_wheelhouse.py must run on Python 3.11")

    _, version = wheel_name_version(project_wheel)
    bundle_name = f"private-gpt-{version}-{extra}-{platform_tag()}"

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        bundle = work / bundle_name
        wheelhouse = bundle / "wheelhouse"
        downloads = work / "downloads"
        wheelhouse.mkdir(parents=True)
        downloads.mkdir()

        locked = work / "locked.txt"
        export_requirements(extra, locked)

        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--require-hashes",
                "--no-deps",
                "-r",
                str(locked),
                "-d",
                str(downloads),
            ]
        )

        built: dict[str, Path] = {}
        for artifact in sorted(downloads.iterdir()):
            if artifact.suffix == ".whl":
                shutil.copy2(artifact, wheelhouse / artifact.name)
                continue
            before = set(wheelhouse.glob("*.whl"))
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "-w",
                    str(wheelhouse),
                    str(artifact),
                ]
            )
            (wheel,) = set(wheelhouse.glob("*.whl")) - before
            built[wheel_name_version(wheel)[0]] = wheel

        shutil.copy2(project_wheel, wheelhouse / project_wheel.name)

        requirements = rewrite_hashes(locked.read_text(), built)
        requirements += (
            f"private-gpt[{extra}]=={version} \\\n"
            f"    --hash=sha256:{sha256(project_wheel)}\n"
        )
        (bundle / "requirements.txt").write_text(requirements)
        (bundle / "INSTALL.txt").write_text(install_text(version, extra))
        (bundle / "SHA256SUMS").write_text(
            "".join(
                f"{sha256(w)}  wheelhouse/{w.name}\n"
                for w in sorted(wheelhouse.glob("*.whl"))
            )
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            archive = out_dir / f"{bundle_name}.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(bundle.rglob("*")):
                    zf.write(path, path.relative_to(work))
        else:
            archive = out_dir / f"{bundle_name}.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(bundle, arcname=bundle_name)

    print(f"Wrote {archive}")
    return archive


def install_text(version: str, extra: str) -> str:
    return f"""PrivateGPT {version} offline bundle ({extra})

Every dependency is pinned to the release lock file and shipped in
wheelhouse/. Installing needs Python 3.11 and no network access.

With pip (verifies every file against requirements.txt):
  python3.11 -m venv private-gpt-venv
  private-gpt-venv/bin/pip install --no-index --find-links wheelhouse \\
    --require-hashes -r requirements.txt
  (Windows: private-gpt-venv\\Scripts\\pip.exe ...)
  Then run:  private-gpt-venv/bin/private-gpt serve

With uv:
  uv tool install --offline --no-index --find-links wheelhouse \\
    --python 3.11 "private-gpt[{extra}]=={version}"
  Then run:  private-gpt serve
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-wheel", required=True, type=Path)
    parser.add_argument("--extra", default="core")
    parser.add_argument("--out-dir", default=Path("dist"), type=Path)
    args = parser.parse_args()
    build(args.project_wheel.resolve(), args.extra, args.out_dir.resolve())


if __name__ == "__main__":
    main()
