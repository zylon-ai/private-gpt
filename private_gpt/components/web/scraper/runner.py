from __future__ import annotations

import importlib.resources
import json
import posixpath
import re
import shlex
from typing import TYPE_CHECKING

from pydantic import BaseModel

from private_gpt.components.sandbox.base import SandboxExecOptions

if TYPE_CHECKING:
    from private_gpt.components.sandbox.base import SandboxSession
    from private_gpt.settings.settings import Settings

SCRIPT_FILENAME = "scrape_script.py"
CONFIG_FILENAME = "scrape_config.json"
RESULTS_FILENAME = "scrape_results.json"
_OUTPUT_FILENAME_TEMPLATE = "page_{index}.html"

# Extra seconds granted to the session exec on top of the script's own budget
# so the script can exit 124 by itself and report a clean timeout.
_EXEC_TIMEOUT_MARGIN_SECONDS = 30

_BROWSERS_MISSING_PATTERN = re.compile(
    r"playwright.*install|browser.*executable|download new browsers", re.IGNORECASE
)


class ScrapeRequest(BaseModel):
    url: str
    output_path: str


class ScrapeConfig(BaseModel):
    """Parameters handed to scrape_script.py as a JSON file.

    ``timeout_seconds`` is per request; the script budgets
    ``timeout_seconds * len(requests)`` for the whole run.
    """

    requests: list[ScrapeRequest]
    timeout_seconds: int
    proxy_server: str | None = None
    proxy_username: str | None = None
    proxy_password: str | None = None
    proxy_bypass: str | None = None
    ignore_https_errors: bool = False


def build_scrape_config(
    settings: Settings, urls: list[str], timeout_seconds: int
) -> ScrapeConfig:
    """Map proxy/SSL settings from ``server.network`` into a scrape config.

    Output paths are relative filenames: the script runs with the session's
    base dir as cwd, so the same config works for host and remote sandboxes
    (whose real filesystem paths differ from the canonical ones).
    Chromium ignores proxy environment variables, so the proxy must be passed
    explicitly for the script to build playwright ``ProxySettings`` from it.
    """
    requests = [
        ScrapeRequest(
            url=url,
            output_path=_OUTPUT_FILENAME_TEMPLATE.format(index=index),
        )
        for index, url in enumerate(urls)
    ]
    config = ScrapeConfig(requests=requests, timeout_seconds=timeout_seconds)

    proxy = settings.server.network.proxy
    if proxy.enabled:
        url_cfg = proxy.https_server or proxy.http_server
        if url_cfg:
            config.proxy_server = f"{url_cfg.scheme}://{url_cfg.host}" + (
                f":{url_cfg.port}" if url_cfg.port else ""
            )
            config.proxy_username = url_cfg.username
            config.proxy_password = url_cfg.password
            config.proxy_bypass = proxy.bypass or None

    ssl = settings.server.network.ssl
    config.ignore_https_errors = not ssl.verify_ssl or bool(ssl.cert_file)
    return config


def _load_script_text() -> str:
    return (
        importlib.resources.files("private_gpt.components.web.scraper")
        .joinpath(SCRIPT_FILENAME)
        .read_text(encoding="utf-8")
    )


async def run_scrape_in_session(
    session: SandboxSession, base_dir: str, config: ScrapeConfig
) -> list[str | Exception]:
    """Write the scrape script + config into the session, run it, read HTML back.

    The script executes with ``base_dir`` as cwd and only relative filenames,
    so canonical-vs-real path differences between backends never leak into it.
    HTML travels through files rather than stdout: session backends reassemble
    or cap stdout, while ``read_file()`` is byte-exact.

    Failures are isolated per request (one bad URL in a batch does not fail
    the others): each slot in the returned list is either the page HTML or
    the exception for that URL. Run-level failures still raise.
    """
    script_path = posixpath.join(base_dir, SCRIPT_FILENAME)
    config_path = posixpath.join(base_dir, CONFIG_FILENAME)
    await session.write_file(script_path, _load_script_text().encode("utf-8"))
    await session.write_file(config_path, config.model_dump_json().encode("utf-8"))

    total_timeout = config.timeout_seconds * max(1, len(config.requests))
    result = await session.exec(
        f"{session.python_executable} {shlex.quote(SCRIPT_FILENAME)}"
        f" {shlex.quote(CONFIG_FILENAME)}",
        SandboxExecOptions(
            cwd=base_dir, timeout=total_timeout + _EXEC_TIMEOUT_MARGIN_SECONDS
        ),
    )

    if result.exit_code == 124:
        raise TimeoutError(
            f"Timeout ({config.timeout_seconds}s) scraping "
            f"{[r.url for r in config.requests]}"
        )
    if result.failed:
        error = result.stderr or result.stdout
        if _BROWSERS_MISSING_PATTERN.search(error):
            raise RuntimeError(
                "Playwright browsers are not installed. "
                "Run `playwright install` and try again."
            )
        raise RuntimeError(
            f"Scrape script failed (exit_code={result.exit_code}): {error}"
        )

    results_raw = json.loads(
        (await session.read_file(posixpath.join(base_dir, RESULTS_FILENAME))).decode(
            "utf-8"
        )
    )
    by_output_path = {entry["output_path"]: entry for entry in results_raw}

    outputs: list[str | Exception] = []
    for request in config.requests:
        entry = by_output_path.get(request.output_path)
        if entry is None:
            outputs.append(
                RuntimeError(f"Scrape script returned no result for {request.url}")
            )
        elif entry.get("error"):
            if entry.get("timeout"):
                outputs.append(
                    TimeoutError(
                        f"Timeout ({config.timeout_seconds}s) scraping {request.url}"
                    )
                )
            else:
                outputs.append(
                    RuntimeError(f"Failed to scrape {request.url}: {entry['error']}")
                )
        else:
            outputs.append(
                (
                    await session.read_file(
                        posixpath.join(base_dir, request.output_path)
                    )
                ).decode("utf-8", errors="replace")
            )
    return outputs
