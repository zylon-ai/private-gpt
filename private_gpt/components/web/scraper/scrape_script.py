"""Standalone Playwright scrape script executed inside a sandbox session.

This file is shipped as package data and copied verbatim into the session
(host workdir or remote sandbox) before execution, so it must not import
anything from private_gpt and must stay compatible with the playwright
version pinned by the sandbox images (1.60.0).

Contract:
- Runs with the session's base dir as cwd; all filenames are relative to it.
- ``argv[1]``: path to a JSON config file with keys:
  ``requests`` (list of ``{"url", "output_path"}``), ``timeout_seconds``
  (per request), ``proxy_server``, ``proxy_username``, ``proxy_password``,
  ``proxy_bypass``, ``ignore_https_errors``.
- One browser serves the whole batch; requests are scraped sequentially and
  failures are isolated per request: each page's rendered HTML (UTF-8) goes
  to its ``output_path`` and a summary goes to ``scrape_results.json`` as
  ``[{"output_path", "error": null|str, "timeout": bool}]``.
- Exit codes: 0 when the run completed (even with per-request errors), 124 on
  global timeout, 1 on any run-level failure with a message on stderr.
"""

import asyncio
import json
import sys
from typing import Any

_EXIT_TIMEOUT = 124
_RESULTS_FILENAME = "scrape_results.json"

_BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-first-run",
]


def _context_options(config: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if config.get("ignore_https_errors"):
        options["ignore_https_errors"] = True
    if config.get("proxy_server"):
        proxy: dict[str, Any] = {"server": config["proxy_server"]}
        for key in ("username", "password", "bypass"):
            value = config.get(f"proxy_{key}")
            if value:
                proxy[key] = value
        options["proxy"] = proxy
    return options


async def _scrape_page(context: Any, request: dict[str, Any]) -> None:
    page = await context.new_page()
    try:
        await page.goto(request["url"])
        await page.wait_for_load_state("domcontentloaded")
        content = await page.content()
    finally:
        await page.close()
    with open(request["output_path"], "w", encoding="utf-8") as fh:
        fh.write(content)


async def _scrape(config: dict[str, Any]) -> None:
    from playwright.async_api import (  # ty:ignore[unresolved-import]
        TimeoutError as PlaywrightTimeoutError,
    )
    from playwright.async_api import (  # ty:ignore[unresolved-import]
        async_playwright,
    )

    timeout_ms = float(config["timeout_seconds"]) * 1000
    results: list[dict[str, Any]] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=_BROWSER_ARGS)
        try:
            context = await browser.new_context(**_context_options(config))
            context.set_default_timeout(timeout_ms)
            for request in config["requests"]:
                entry: dict[str, Any] = {
                    "output_path": request["output_path"],
                    "error": None,
                    "timeout": False,
                }
                try:
                    await _scrape_page(context, request)
                except Exception as exc:
                    entry["error"] = str(exc) or exc.__class__.__name__
                    entry["timeout"] = isinstance(
                        exc, PlaywrightTimeoutError | asyncio.TimeoutError
                    )
                results.append(entry)
        finally:
            await browser.close()
    with open(_RESULTS_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(results, fh)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        config = json.load(fh)

    total_timeout = float(config["timeout_seconds"]) * max(1, len(config["requests"]))
    try:
        asyncio.run(asyncio.wait_for(_scrape(config), timeout=total_timeout))
    except TimeoutError:
        print(f"Scrape timed out after {total_timeout:.0f}s", file=sys.stderr)
        sys.exit(_EXIT_TIMEOUT)
    except Exception as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
