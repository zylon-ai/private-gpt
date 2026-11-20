import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import AnyHttpUrl, BaseModel

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from httpx2 import URL, Response

_PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"


class ParallelSearchHit(BaseModel):
    url: AnyHttpUrl
    title: str | None = None
    publish_date: str | None = None
    excerpts: list[str]


class ParallelSearchResponse(BaseModel):
    results: list[ParallelSearchHit]


class ParallelSearchProvider(BaseWebSearchProvider):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings.parallel
        self._network = settings.server.network

    async def validate(self) -> None:
        if self._network.offline_mode:
            raise ValueError("Parallel web search is unavailable in offline mode")

        try:
            import mcp.client.streamable_http  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                format_missing_dependency_message(
                    "Parallel web search", extras="tool-mcp"
                )
            ) from exc

    async def make_query(
        self, query: str, num_links: int, **kwargs: Any
    ) -> list[WebSearchResult]:
        from httpx2 import AsyncClient
        from mcp import ClientSession, MCPError
        from mcp.client.streamable_http import streamable_http_client
        from mcp.types import CallToolResult

        class ParallelMCPClient(AsyncClient):
            async def delete(self, url: "URL | str", **kwargs: Any) -> "Response":
                # Bound the whole session close, including a streaming response.
                async with asyncio.timeout(1):
                    return await super().delete(url, **kwargs)

        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")

        headers = {"User-Agent": "PrivateGPT"}
        if self._settings.api_key and self._settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self._settings.api_key.strip()}"

        result = None
        error = None
        async with (
            ParallelMCPClient(
                headers=headers,
                timeout=self._settings.timeout,
                trust_env=self._network.proxy.enabled,
                follow_redirects=False,
            ) as client,
            streamable_http_client(_PARALLEL_MCP_URL, http_client=client) as (
                reader,
                writer,
            ),
            ClientSession(reader, writer) as session,
        ):
            try:
                async with asyncio.timeout(self._settings.timeout):
                    await session.initialize()
                    await session.list_tools()
                    result = await session.call_tool(
                        "web_search",
                        {"objective": query, "search_queries": [query]},
                    )
            except (MCPError, TimeoutError, RuntimeError) as exc:
                # Let the SDK close its task groups before raising.
                error = exc

        if error is not None:
            if isinstance(error, TimeoutError):
                raise TimeoutError("Parallel web search timed out") from error
            raise RuntimeError(f"Parallel web search failed: {error}") from error
        if not isinstance(result, CallToolResult) or result.is_error:
            raise RuntimeError("Parallel web search returned a tool error")

        response = ParallelSearchResponse.model_validate(result.structured_content)
        return [
            WebSearchResult(
                idx=index,
                title=hit.title or str(hit.url),
                url=str(hit.url),
                description="\n\n".join(hit.excerpts),
                age=hit.publish_date,
            )
            for index, hit in enumerate(response.results[: max(0, num_links)], start=1)
        ]

    async def close(self) -> None:
        # Each search owns its HTTP client and MCP session.
        pass
