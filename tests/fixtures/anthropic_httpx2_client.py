from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx2

if TYPE_CHECKING:
    from starlette.testclient import TestClient

Handler = Callable[[httpx2.Request], httpx2.Response]


def _forward_to_test_client(test_client: TestClient) -> Handler:
    """Translate an SDK (httpx2) request into a Starlette TestClient round-trip."""

    def build_response(request: httpx2.Request) -> httpx2.Response:
        starlette_request = test_client.build_request(
            method=request.method,
            url=request.url.path,
            headers=dict(request.headers),
            content=request.content,
            params=dict(request.url.params),
        )
        response = test_client.send(starlette_request)
        content_type = response.headers.get("Content-Type", "")
        headers = dict(response.headers)

        if "text/event-stream" in content_type:
            raw_events = [
                (item + "\n\n").encode("utf-8")
                for item in response.text.split("\n\n")
                if item.strip()
            ]
            return httpx2.Response(
                status_code=response.status_code,
                headers=headers,
                stream=httpx2.ByteStream(b"".join(raw_events)),
            )

        if "application/json" in content_type:
            return httpx2.Response(
                status_code=response.status_code,
                headers=headers,
                json=response.json(),
            )

        return httpx2.Response(
            status_code=response.status_code,
            headers=headers,
            content=response.content,
        )

    return build_response


def create_mock_http_client(
    test_client: TestClient | None = None,
    *,
    is_async: bool = False,
    handler: Handler | None = None,
) -> httpx2.Client | httpx2.AsyncClient:
    """Build the httpx2 client expected by anthropic>=1.

    Anthropic 1.x moved its HTTP layer from ``httpx`` to ``httpx2``. Tests
    that inject ``httpx.Client`` (or rely on pytest-httpx, which only patches
    ``httpx``) fail with ``TypeError: Invalid type for url ... got httpx2.URL``.
    """
    if handler is None:
        if test_client is None:
            raise ValueError("test_client is required unless handler is provided")
        handler = _forward_to_test_client(test_client)

    transport = httpx2.MockTransport(handler)
    if is_async:
        return httpx2.AsyncClient(transport=transport, base_url="http://testserver")
    return httpx2.Client(transport=transport, base_url="http://testserver")
