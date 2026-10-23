#!/usr/bin/env python3
"""Generate the Homebrew formula for private-gpt."""

from __future__ import annotations

import argparse
from pathlib import Path


def render_formula(
    *,
    version: str,
    source_url: str,
    sha256: str,
    homepage: str,
    package_files_url: str,
    extra: str,
) -> str:
    return f"""class PrivateGpt < Formula
  desc "Private self-hosted AI API server"
  homepage "{homepage}"
  url "{source_url}"
  sha256 "{sha256}"
  license "Apache-2.0"

  depends_on "python@3.11"
  depends_on "uv"

  def install
    (bin/"private-gpt").write <<~SH
      #!/bin/bash
      exec "#{{Formula["uv"].opt_bin}}/uv" tool run --python "#{{Formula["python@3.11"].opt_bin}}/python3.11" --find-links "{package_files_url}" --from "private-gpt[{extra}]=={version}" private-gpt "$@"
    SH
  end

  test do
    script = (bin/"private-gpt").read
    assert_match "private-gpt[{extra}]=={version}", script
    assert_match "{package_files_url}", script
    assert_match Formula["uv"].opt_bin.realpath.to_s, script
  end
end
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--homepage",
        default="https://github.com/zylon-ai/private-gpt",
    )
    parser.add_argument("--extra", default="core")
    parser.add_argument("--package-files-url", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formula = render_formula(
        version=args.version,
        source_url=args.source_url,
        sha256=args.sha256,
        homepage=args.homepage,
        package_files_url=args.package_files_url,
        extra=args.extra,
    )
    Path(args.output).write_text(formula)


if __name__ == "__main__":
    main()
