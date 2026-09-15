"""URL loading must be bounded (timeout, size) and must not accept error pages."""

import io
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from private_gpt.server.ingest.uri_loader import load_file_from_uri


def _response(status: int, chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.iter_content.return_value = iter(chunks)
    response.content = b"".join(chunks)
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_url_loader_rejects_http_error_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    get = MagicMock(return_value=_response(404, [b"<html>not found</html>"]))
    monkeypatch.setattr("requests.get", get)

    with pytest.raises(requests.HTTPError):
        load_file_from_uri("https://files.example.com/deck.pptx")


def test_url_loader_passes_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    get = MagicMock(return_value=_response(200, [b"%PDF-1.4"]))
    monkeypatch.setattr("requests.get", get)

    data = load_file_from_uri("https://files.example.com/doc.pdf")

    assert data.read() == b"%PDF-1.4"
    timeout: Any = get.call_args.kwargs.get("timeout")
    assert timeout is not None, "requests.get called without a timeout"


def test_url_loader_enforces_size_limit_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(200, [b"x" * 1024] * 8)
    get = MagicMock(return_value=response)
    monkeypatch.setattr("requests.get", get)

    with pytest.raises(ValueError, match="exceeds"):
        load_file_from_uri("https://files.example.com/big.bin", max_bytes=2048)

    # The body must be streamed and abandoned, not fully buffered first.
    assert get.call_args.kwargs.get("stream") is True


def test_url_loader_returns_body_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    get = MagicMock(return_value=_response(200, [b"abc", b"def"]))
    monkeypatch.setattr("requests.get", get)

    data = load_file_from_uri("https://files.example.com/ok.txt", max_bytes=6)

    assert isinstance(data, io.BytesIO)
    assert data.read() == b"abcdef"
