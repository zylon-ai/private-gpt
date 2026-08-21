import base64
from functools import partial
from urllib.parse import parse_qs

import httpx2
import pytest
from mcp.client.auth import OAuthFlowError
from mcp.shared.auth import OAuthClientMetadata

from private_gpt.server.mcp._runtime import (
    MISSING_ACCESS_TOKEN,
    HeadlessOAuthClientProvider,
    RequestOAuthTokenStorage,
    _check_auth,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("access_token", "auth_method", "client_secret", "expected_auth_method"),
    [
        (None, None, "client-secret", "client_secret_basic"),
        (
            "expired-access",
            "client_secret_basic",
            "client-secret",
            "client_secret_basic",
        ),
        (
            "expired-access",
            "client_secret_post",
            "client-secret",
            "client_secret_post",
        ),
        ("expired-access", "none", None, "none"),
    ],
)
async def test_headless_oauth_discovers_and_refreshes(
    access_token: str | None,
    auth_method: str | None,
    client_secret: str | None,
    expected_auth_method: str,
) -> None:
    requests: list[tuple[str, str, str | None, str]] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = (await request.aread()).decode()
        url = str(request.url)
        requests.append(
            (request.method, url, request.headers.get("authorization"), body)
        )
        if url == "https://resource.example.com/mcp":
            if request.headers.get("authorization") == "Bearer access-after":
                return httpx2.Response(200, json={"ok": True})
            return httpx2.Response(
                401,
                headers={
                    "WWW-Authenticate": (
                        'Bearer resource_metadata="https://resource.example.com/'
                        '.well-known/oauth-protected-resource/mcp"'
                    )
                },
            )
        if url == (
            "https://resource.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx2.Response(
                200,
                json={
                    "resource": "https://resource.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        if url == "https://auth.example.com/.well-known/oauth-authorization-server":
            return httpx2.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token"],
                },
            )
        if url == "https://auth.example.com/token":
            return httpx2.Response(
                200,
                json={
                    "access_token": "access-after",
                    "token_type": "Bearer",
                    "refresh_token": "refresh-after",
                },
            )
        return httpx2.Response(404)

    storage = RequestOAuthTokenStorage(
        access_token=access_token,
        refresh_token="refresh-before",
        client_id="client-id",
        client_secret=client_secret,
        token_endpoint_auth_method=auth_method,
    )
    assert storage.refreshed_tokens is None
    auth = HeadlessOAuthClientProvider(
        server_url="https://resource.example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=["http://127.0.0.1/mcp-oauth"]
        ),
        storage=storage,
    )

    async with httpx2.AsyncClient(
        auth=auth,
        transport=httpx2.MockTransport(handler),
    ) as client:
        response = await client.post("https://resource.example.com/mcp", content=b"{}")

    assert response.status_code == 200
    assert requests[0][2] == f"Bearer {access_token or MISSING_ACCESS_TOKEN}"
    assert [request[1] for request in requests] == [
        "https://resource.example.com/mcp",
        "https://resource.example.com/.well-known/oauth-protected-resource/mcp",
        "https://auth.example.com/.well-known/oauth-authorization-server",
        "https://auth.example.com/token",
        "https://resource.example.com/mcp",
    ]
    token_request = requests[3]
    token_data = parse_qs(token_request[3])
    assert token_data["refresh_token"] == ["refresh-before"]
    if expected_auth_method == "client_secret_basic":
        credentials = base64.b64encode(b"client-id:client-secret").decode()
        assert token_request[2] == f"Basic {credentials}"
        assert "client_secret" not in token_data
    elif expected_auth_method == "client_secret_post":
        assert token_request[2] is None
        assert token_data["client_secret"] == ["client-secret"]
    else:
        assert token_request[2] is None
        assert "client_secret" not in token_data
    assert requests[-1][2] == "Bearer access-after"
    assert storage.refreshed_tokens == (
        "access-after",
        "refresh-after",
        "refresh-before",
    )


@pytest.mark.asyncio
async def test_auth_discovery_failure_marks_refresh_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == "https://resource.example.com/mcp":
            return httpx2.Response(
                401,
                headers={
                    "WWW-Authenticate": (
                        'Bearer resource_metadata="https://resource.example.com/'
                        '.well-known/oauth-protected-resource/mcp"'
                    )
                },
            )
        if str(request.url) == (
            "https://resource.example.com/.well-known/oauth-protected-resource/mcp"
        ):
            return httpx2.Response(
                200,
                json={
                    "resource": "https://different.example.com/mcp",
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        return httpx2.Response(500)

    storage = RequestOAuthTokenStorage(
        access_token="expired-access",
        refresh_token="refresh-before",
        client_id="client-id",
        client_secret=None,
    )
    auth = HeadlessOAuthClientProvider(
        server_url="https://resource.example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=["http://127.0.0.1/mcp-oauth"]
        ),
        storage=storage,
    )
    monkeypatch.setattr(
        "private_gpt.server.mcp._runtime.httpx2.AsyncClient",
        partial(httpx2.AsyncClient, transport=httpx2.MockTransport(handler)),
    )

    with pytest.raises(OAuthFlowError):
        await _check_auth("https://resource.example.com/mcp", {}, auth)

    assert storage.refresh_attempted is True
