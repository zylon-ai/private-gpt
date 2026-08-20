import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx2
    from mcp import ClientSession, MCPError
    from mcp.client.auth import OAuthClientProvider, OAuthFlowError, TokenStorage
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from mcp.types import (
        AudioContent,
        CallToolResult,
        ImageContent,
        ListToolsResult,
        TextContent,
    )
else:
    import httpx2
    from mcp import ClientSession, MCPError
    from mcp.client.auth import OAuthClientProvider, OAuthFlowError, TokenStorage
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )
    from mcp.shared.auth import (
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from mcp.types import (
        AudioContent,
        CallToolResult,
        ImageContent,
        ListToolsResult,
        TextContent,
    )

logger = logging.getLogger(__name__)
MISSING_ACCESS_TOKEN = "mcp-missing-access-token"

__all__ = [
    "AudioContent",
    "CallToolResult",
    "ClientSession",
    "ImageContent",
    "ListToolsResult",
    "MCPError",
    "PersistentMCPClient",
    "TextContent",
]


class SessionError(Exception):
    """Custom exception for session-related errors."""


def _prefer_sse(url: str) -> bool:
    """Heuristic for legacy SSE endpoints vs streamable HTTP."""
    path = urlparse(url).path.lower()
    return path.endswith("/sse") or "/sse/" in path


class RequestOAuthTokenStorage(TokenStorage):
    """Request-scoped OAuth storage initialized from the MCP request."""

    def __init__(
        self,
        *,
        access_token: str | None,
        refresh_token: str,
        client_id: str,
        client_secret: str | None,
        token_endpoint_auth_method: str | None = None,
    ) -> None:
        # The placeholder represents a missing access token, not a refresh.
        self._tokens = OAuthToken(
            access_token=access_token or MISSING_ACCESS_TOKEN,
            refresh_token=refresh_token,
        )
        self._refreshed_tokens: tuple[str, str, str] | None = None
        self._client_info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=token_endpoint_auth_method
            or ("client_secret_basic" if client_secret else "none"),
            redirect_uris=[],
        )
        self.refresh_attempted = False

    @property
    def refreshed_tokens(self) -> tuple[str, str, str] | None:
        return self._refreshed_tokens

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        previous_refresh_token = self._tokens.refresh_token
        assert previous_refresh_token is not None
        if tokens.refresh_token is None:
            tokens.refresh_token = previous_refresh_token
        self._tokens = tokens
        self._refreshed_tokens = (
            tokens.access_token,
            tokens.refresh_token or previous_refresh_token,
            previous_refresh_token,
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class HeadlessOAuthClientProvider(OAuthClientProvider):
    """OAuth provider that discovers endpoints and only permits refresh."""

    async def _refresh_token(self) -> httpx2.Request:
        storage = self.context.storage
        if isinstance(storage, RequestOAuthTokenStorage):
            storage.refresh_attempted = True
        return await super()._refresh_token()

    async def _perform_authorization(self) -> httpx2.Request:
        if not self.context.can_refresh_token():
            raise RuntimeError("MCP OAuth refresh requires a refresh token")
        return await self._refresh_token()


async def _check_auth(
    url: str,
    headers: dict[str, Any],
    auth: httpx2.Auth | None = None,
) -> None:
    """Do a pre-flight POST to detect 401/403 before entering the MCP transport."""
    oauth_storage = (
        auth.context.storage if isinstance(auth, HeadlessOAuthClientProvider) else None
    )
    try:
        async with httpx2.AsyncClient(
            auth=auth,
            follow_redirects=True,
            timeout=10.0,
        ) as client:
            response = await client.post(url, headers=headers, content=b"{}")
    except OAuthFlowError:
        if isinstance(oauth_storage, RequestOAuthTokenStorage):
            oauth_storage.refresh_attempted = True
        raise
    if response.status_code in (401, 403):
        if isinstance(oauth_storage, RequestOAuthTokenStorage):
            oauth_storage.refresh_attempted = True
        response.raise_for_status()


class PersistentMCPClient:
    """Native MCP 2.x client with persistent session recovery.

    Transport is implemented directly against the official ``mcp`` package.
    Tool schemas are never rewritten through third-party converters.
    """

    def __init__(
        self,
        command_or_url: str,
        *,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        refresh_token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        token_endpoint_auth_method: str | None = None,
        **_: Any,
    ) -> None:
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.headers = headers or {}
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self.oauth_storage = (
            RequestOAuthTokenStorage(
                access_token=self.headers.get("Authorization", "").removeprefix(
                    "Bearer "
                )
                or None,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                token_endpoint_auth_method=token_endpoint_auth_method,
            )
            if refresh_token and client_id
            else None
        )
        self.auth = (
            HeadlessOAuthClientProvider(
                server_url=command_or_url,
                client_metadata=OAuthClientMetadata(
                    redirect_uris=["http://127.0.0.1/mcp-oauth"]
                ),
                storage=self.oauth_storage,
            )
            if self.oauth_storage
            else None
        )

        self._persistent_session: ClientSession | None = None
        self._session_context: AsyncExitStack | None = None
        self._session_lock = asyncio.Lock()
        self._closed = False

    @property
    def refresh_attempted(self) -> bool:
        return bool(self.oauth_storage and self.oauth_storage.refresh_attempted)

    @property
    def refreshed_tokens(self) -> tuple[str, str, str] | None:
        return self.oauth_storage.refreshed_tokens if self.oauth_storage else None

    async def _create_session(self) -> ClientSession:
        """Create and initialize a new MCP session, keeping resources open."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            url = urlparse(self.command_or_url)
            scheme = url.scheme

            if scheme in ("http", "https"):
                if _prefer_sse(self.command_or_url):
                    read_stream, write_stream = await stack.enter_async_context(
                        sse_client(
                            self.command_or_url,
                            headers=self.headers or None,
                            timeout=self.timeout,
                            sse_read_timeout=self.sse_read_timeout,
                            auth=self.auth,
                        )
                    )
                else:
                    await _check_auth(self.command_or_url, self.headers, self.auth)
                    http_client = create_mcp_http_client(auth=self.auth)
                    if self.headers:
                        http_client.headers.update(self.headers)
                    await stack.enter_async_context(http_client)
                    read_stream, write_stream = await stack.enter_async_context(
                        streamable_http_client(
                            self.command_or_url,
                            http_client=http_client,
                        )
                    )
            else:
                server_parameters = StdioServerParameters(
                    command=self.command_or_url,
                    args=self.args,
                    env=self.env or None,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server_parameters)
                )

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self.timeout,
                )
            )
            await session.initialize()
            self._session_context = stack
            self._persistent_session = session
            return session
        except Exception:
            await stack.aclose()
            self._session_context = None
            self._persistent_session = None
            raise

    @asynccontextmanager
    async def _run_session(self) -> AsyncIterator[ClientSession]:
        """Provide a persistent session with automatic recovery."""
        if self._closed:
            raise SessionError("Client has been closed")

        async with self._session_lock:
            if self._persistent_session is not None:
                try:
                    yield self._persistent_session
                    return
                except (MCPError, ConnectionError, TimeoutError, OSError) as e:
                    logger.warning("Session error: %s, attempting recovery", e)
                    await self._reset_session()

            last_exception: Exception | None = None
            for attempt in range(self._max_retries):
                try:
                    session = await self._create_session()
                    logger.info(
                        "Session created successfully for %s",
                        self.command_or_url,
                    )
                    yield session
                    return
                except httpx2.HTTPStatusError:
                    await self._reset_session()
                    raise
                except (MCPError, ConnectionError, TimeoutError, OSError) as e:
                    last_exception = e
                    logger.warning(
                        "Session creation failed (attempt %s/%s): %s",
                        attempt + 1,
                        self._max_retries,
                        e,
                    )
                    await self._reset_session()
                    if attempt < self._max_retries - 1:
                        delay = self._retry_delay * (2**attempt)
                        logger.info("Retrying in %.2f seconds...", delay)
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(
                        "Unexpected error creating session: %s", e, exc_info=True
                    )
                    await self._reset_session()
                    raise

            raise SessionError(
                f"Session creation failed after {self._max_retries} attempts: "
                f"{last_exception}"
            ) from last_exception

    async def _reset_session(self) -> None:
        if self._session_context is not None:
            try:
                await self._session_context.aclose()
            except Exception as e:
                logger.warning("Error during session reset: %s", e)
            finally:
                self._session_context = None
                self._persistent_session = None
        else:
            self._persistent_session = None

    async def list_tools(self) -> ListToolsResult:
        async with self._run_session() as session:
            return await session.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        async with self._run_session() as session:
            result = await session.call_tool(name=name, arguments=arguments)
            if not isinstance(result, CallToolResult):
                raise TypeError(
                    f"Unexpected MCP call_tool result type: {type(result)!r}"
                )
            return result

    async def health_check(self) -> bool:
        try:
            await self.list_tools()
            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    async def close(self) -> None:
        async with self._session_lock:
            self._closed = True
            if self._session_context is not None:
                try:
                    await self._session_context.aclose()
                    logger.info("Session closed for %s", self.command_or_url)
                except Exception as e:
                    logger.error("Error closing session: %s", e)
                finally:
                    self._session_context = None
                    self._persistent_session = None

    async def __aenter__(self) -> "PersistentMCPClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
