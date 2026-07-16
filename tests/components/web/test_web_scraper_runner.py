import json
from unittest.mock import MagicMock

import pytest

from private_gpt.components.sandbox.base import SandboxExecutionResult
from private_gpt.components.web.scraper.runner import (
    CONFIG_FILENAME,
    RESULTS_FILENAME,
    SCRIPT_FILENAME,
    ScrapeConfig,
    ScrapeRequest,
    build_scrape_config,
    run_scrape_in_session,
)
from private_gpt.settings.settings import Settings


def _results_json(*entries: dict) -> bytes:
    return json.dumps(
        [
            {"output_path": e["output_path"], "error": None, "timeout": False} | e
            for e in entries
        ]
    ).encode()


class _FakeSession:
    """Duck-typed SandboxSession recording writes and serving canned files."""

    python_executable = "python"

    def __init__(
        self,
        exec_result: SandboxExecutionResult,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.exec_result = exec_result
        self.files = files or {}
        self.written: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.cwds: list[str | None] = []

    async def exec(self, command, opts=None):
        self.commands.append(command)
        self.cwds.append(opts.cwd if opts else None)
        return self.exec_result

    async def write_file(self, path: str, content: bytes) -> None:
        self.written[path] = content

    async def read_file(self, path: str) -> bytes:
        return self.files[path]


def _success(exit_code: int = 0) -> SandboxExecutionResult:
    return SandboxExecutionResult(success=exit_code == 0, exit_code=exit_code)


def _config(output_path: str = "page_0.html") -> ScrapeConfig:
    return ScrapeConfig(
        requests=[ScrapeRequest(url="https://example.com", output_path=output_path)],
        timeout_seconds=10,
    )


async def test_success_writes_script_and_config_and_reads_html() -> None:
    session = _FakeSession(
        _success(),
        files={
            "/work/page_0.html": b"<html>hi</html>",
            "/work/" + RESULTS_FILENAME: _results_json({"output_path": "page_0.html"}),
        },
    )

    htmls = await run_scrape_in_session(session, "/work/", _config())

    assert htmls == ["<html>hi</html>"]
    assert "/work/" + SCRIPT_FILENAME in session.written
    assert "/work/" + CONFIG_FILENAME in session.written
    script = session.written["/work/" + SCRIPT_FILENAME].decode()
    assert "playwright" in script
    # relative filenames + cwd, so host/sandbox path differences never leak in
    assert session.commands == [f"python {SCRIPT_FILENAME} {CONFIG_FILENAME}"]
    assert session.cwds == ["/work/"]


async def test_batch_failures_are_isolated_per_request() -> None:
    session = _FakeSession(
        _success(),
        files={
            "/work/page_0.html": b"<html>ok</html>",
            "/work/" + RESULTS_FILENAME: _results_json(
                {"output_path": "page_0.html"},
                {"output_path": "page_1.html", "error": "net::ERR_FAILED"},
                {"output_path": "page_2.html", "error": "nav timeout", "timeout": True},
            ),
        },
    )
    config = ScrapeConfig(
        requests=[
            ScrapeRequest(url="https://ok.com", output_path="page_0.html"),
            ScrapeRequest(url="https://bad.com", output_path="page_1.html"),
            ScrapeRequest(url="https://slow.com", output_path="page_2.html"),
        ],
        timeout_seconds=10,
    )

    results = await run_scrape_in_session(session, "/work/", config)

    assert results[0] == "<html>ok</html>"
    assert isinstance(results[1], RuntimeError)
    assert "net::ERR_FAILED" in str(results[1])
    assert isinstance(results[2], TimeoutError)


async def test_exit_124_raises_timeout() -> None:
    session = _FakeSession(_success(exit_code=124))
    with pytest.raises(TimeoutError):
        await run_scrape_in_session(session, "/work/", _config())


async def test_failure_raises_runtime_error_with_stderr() -> None:
    session = _FakeSession(
        SandboxExecutionResult(success=False, exit_code=1, stderr="boom")
    )
    with pytest.raises(RuntimeError, match="boom"):
        await run_scrape_in_session(session, "/work/", _config())


async def test_missing_browsers_get_actionable_hint() -> None:
    session = _FakeSession(
        SandboxExecutionResult(
            success=False,
            exit_code=1,
            stderr="Executable doesn't exist, run playwright install",
        )
    )
    with pytest.raises(RuntimeError, match="playwright install"):
        await run_scrape_in_session(session, "/work/", _config())


def _network_settings(
    *,
    proxy_enabled: bool = False,
    verify_ssl: bool = True,
    cert_file: str | None = None,
) -> Settings:
    settings = MagicMock(spec=Settings)
    settings.server = MagicMock()
    proxy = MagicMock()
    proxy.enabled = proxy_enabled
    if proxy_enabled:
        url = MagicMock()
        url.scheme = "http"
        url.host = "proxy.local"
        url.port = 3128
        url.username = "user"
        url.password = "secret"
        proxy.https_server = url
        proxy.http_server = None
        proxy.bypass = "*.internal"
    settings.server.network.proxy = proxy
    ssl = MagicMock()
    ssl.verify_ssl = verify_ssl
    ssl.cert_file = cert_file
    settings.server.network.ssl = ssl
    return settings


def test_build_scrape_config_maps_proxy_and_ssl() -> None:
    settings = _network_settings(proxy_enabled=True, verify_ssl=False)

    config = build_scrape_config(settings, ["https://example.com"], 10)

    assert config.proxy_server == "http://proxy.local:3128"
    assert config.proxy_username == "user"
    assert config.proxy_password == "secret"
    assert config.proxy_bypass == "*.internal"
    assert config.ignore_https_errors is True
    assert config.requests[0].output_path == "page_0.html"


def test_build_scrape_config_defaults_without_proxy() -> None:
    settings = _network_settings()

    config = build_scrape_config(settings, ["https://a.com", "https://b.com"], 10)

    assert config.proxy_server is None
    assert config.ignore_https_errors is False
    assert [r.output_path for r in config.requests] == [
        "page_0.html",
        "page_1.html",
    ]


def test_custom_ca_cert_implies_ignore_https_errors() -> None:
    settings = _network_settings(cert_file="/certs/ca.pem")
    config = build_scrape_config(settings, ["https://example.com"], 10)
    assert config.ignore_https_errors is True
