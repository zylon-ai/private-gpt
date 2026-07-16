# ruff: noqa: I001

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_index.core.tools import FunctionTool

    class ClientSession:
        ...

    class McpError(Exception):
        ...

    class TextContent:
        text: str

    class ImageContent:
        data: str
        mimeType: str

    class AudioContent:
        data: str
        mimeType: str

    class CallToolResult:
        content: list[object]

    async def aget_tools_from_mcp_url(
        *args: object, **kwargs: object
    ) -> list["FunctionTool"]:
        ...

    class PersistentMCPClient:
        command_or_url: str

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            ...

        async def list_tools(self) -> object:
            ...

        async def close(self) -> None:
            ...

else:
    from llama_index.tools.mcp import BasicMCPClient  # type: ignore[import-untyped]
    from llama_index.tools.mcp import aget_tools_from_mcp_url  # type: ignore[import-untyped]
    from mcp import ClientSession, McpError  # type: ignore[import-not-found]
    from mcp.types import (  # type: ignore[import-not-found]
        AudioContent,
        CallToolResult,
        ImageContent,
        TextContent,
    )

    logger = logging.getLogger(__name__)

    __all__ = [
        "AudioContent",
        "CallToolResult",
        "ClientSession",
        "ImageContent",
        "McpError",
        "PersistentMCPClient",
        "TextContent",
        "aget_tools_from_mcp_url",
    ]

    class SessionError(Exception):
        """Custom exception for session-related errors."""

        pass

    class PersistentMCPClient(BasicMCPClient):
        """MCP Client that maintains a persistent session with automatic recovery."""

        def __init__(
            self,
            *args: Any,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._persistent_session: ClientSession | None = None
            self._session_context: AsyncExitStack | None = None
            self._session_lock = asyncio.Lock()
            self._max_retries = max_retries
            self._retry_delay = retry_delay
            self._closed = False
            self._retry_count = 0

        @asynccontextmanager
        async def _run_session(self) -> AsyncIterator[ClientSession]:
            """Override to provide persistent session with retry logic."""
            if self._closed:
                raise SessionError("Client has been closed")

            async with self._session_lock:
                if self._persistent_session is not None:
                    try:
                        yield self._persistent_session
                        self._retry_count = 0
                        return
                    except (McpError, ConnectionError, TimeoutError, OSError) as e:
                        logger.warning("Session error: %s, attempting recovery", e)
                        await self._reset_session()

                last_exception: Exception | None = None

                for attempt in range(self._max_retries):
                    try:
                        self._session_context = AsyncExitStack()
                        session_cm = super()._run_session()
                        session = await self._session_context.enter_async_context(
                            session_cm
                        )
                        self._persistent_session = session

                        logger.info(
                            "Session created successfully for %s", self.command_or_url
                        )

                        yield session
                        self._retry_count = 0
                        return

                    except (McpError, ConnectionError, TimeoutError, OSError) as e:
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

                error_msg = (
                    "Session creation failed after "
                    f"{self._max_retries} attempts: {last_exception}"
                )
                logger.error(error_msg)
                raise SessionError(error_msg) from last_exception

        async def _reset_session(self) -> None:
            """Reset the session by closing and clearing references."""
            if self._session_context is not None:
                try:
                    await self._session_context.aclose()
                except Exception as e:
                    logger.warning("Error during session reset: %s", e)
                finally:
                    self._session_context = None
                    self._persistent_session = None

        async def health_check(self) -> bool:
            """Check if the session is healthy by attempting to list tools."""
            try:
                await self.list_tools()
                return True
            except Exception as e:
                logger.warning("Health check failed: %s", e)
                return False

        async def close(self) -> None:
            """Close the persistent session and prevent future operations."""
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
