import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp import ClientError, ClientResponse
from injector import inject, singleton

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider
from private_gpt.settings.settings import Settings
from private_gpt.utils.retry import retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.specstream.ai"
_WEB_SEARCH_ENDPOINT = "/v1/web-search"
_MAX_QUERY_LENGTH = 2000
_MIN_NUM_RESULTS = 1
_MAX_NUM_RESULTS = 100
_MAX_PAGE = 10
_VALID_SPEEDS = ("fast", "default")
_MAX_RETRIES = 3
_DELAY = 1.1
_BACKOFF = 2
_JITTER = (1.0, 5.0)
_GAP = 0.1


class SpecStreamRateLimitError(Exception):
    """Raised when SpecStream API rate limits or reports temporary capacity issues."""


class SpecStreamQuotaError(Exception):
    """Raised when SpecStream account balance or key spend limit prevents a search."""


class SpecStreamServerError(Exception):
    """Raised on transient SpecStream API server errors."""


@singleton
class SpecStreamSearchProvider(BaseWebSearchProvider):
    """Web search provider backed by the SpecStream Search API."""

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = self._settings.specstream.api_key
        self._speed = self._settings.specstream.speed
        self._rate_limit = self._settings.specstream.rate_limit
        self._timeout = self._settings.specstream.timeout
        self._session: aiohttp.ClientSession | None = None
        self._rate_limit_lock: asyncio.Lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    async def validate(self) -> None:
        """Verify that a SpecStream API key is configured."""
        if not self._api_key or not self._api_key.strip():
            raise ValueError("SpecStream Search API key is not configured")

    async def make_query(
        self,
        query: str,
        num_links: int = 10,
        offset: int = 0,
        safesearch: bool = True,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        """Execute a SpecStream web search query and return standardized results."""
        normalized_query, normalized_num, page, speed = self._validate_query_params(
            query=query,
            num_links=num_links,
            offset=offset,
            page=kwargs.get("page"),
            speed=kwargs.get("speed"),
        )

        response_data = await self._execute_with_retry(
            query=normalized_query,
            num_links=normalized_num,
            page=page,
            speed=speed,
            safesearch=safesearch,
            language=language,
        )

        return await asyncio.to_thread(self._parse_response, response_data)

    def _validate_query_params(
        self,
        query: str,
        num_links: int,
        offset: int,
        page: Any = None,
        speed: Any = None,
    ) -> tuple[str, int, int, str]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        stripped_query = query.strip()
        if len(stripped_query) > _MAX_QUERY_LENGTH:
            raise ValueError(
                f"Search query cannot exceed {_MAX_QUERY_LENGTH} characters"
            )

        normalized_num = max(_MIN_NUM_RESULTS, min(_MAX_NUM_RESULTS, num_links))
        if normalized_num != num_links:
            logger.warning(
                "num_links %s outside valid range [%s,%s], clamped to %s",
                num_links,
                _MIN_NUM_RESULTS,
                _MAX_NUM_RESULTS,
                normalized_num,
            )

        if offset < 0:
            raise ValueError("Offset cannot be negative")

        resolved_page = (
            int(page) if page is not None else (offset // normalized_num) + 1
        )
        normalized_page = max(1, min(_MAX_PAGE, resolved_page))
        if normalized_page != resolved_page:
            logger.warning(
                "page %s outside valid range [1,%s], clamped to %s",
                resolved_page,
                _MAX_PAGE,
                normalized_page,
            )

        resolved_speed = speed if speed is not None else self._speed
        if resolved_speed not in _VALID_SPEEDS:
            raise ValueError(
                f"Unsupported SpecStream speed {resolved_speed!r}. "
                f"Expected one of: {', '.join(_VALID_SPEEDS)}"
            )

        return stripped_query, normalized_num, normalized_page, resolved_speed

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            enabled_proxy = self._settings.server.network.proxy.enabled
            self._session = aiohttp.ClientSession(trust_env=enabled_proxy)
        return self._session

    async def _wait_for_rate_limit(self) -> None:
        if self._rate_limit <= 0:
            return

        async with self._rate_limit_lock:
            min_interval = (1.0 + _GAP) / self._rate_limit
            now = asyncio.get_running_loop().time()
            time_since_last = now - self._last_request_time
            wait_time = max(0.0, min_interval - time_since_last)
            self._last_request_time = max(now, self._last_request_time) + min_interval

        if wait_time > 0:
            await asyncio.sleep(wait_time)

    def _build_request_params(
        self,
        query: str,
        num_links: int,
        page: int,
        speed: str,
        safesearch: bool,
        language: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "num": num_links,
            "speed": speed,
            "safe": "strict" if safesearch else "off",
        }
        if page > 1:
            params["page"] = page
        if language:
            params["language"] = language
        return params

    def _build_request_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Authorization": f"Bearer {self._api_key.strip()}",
        }

    @staticmethod
    def _extract_error_message(payload: Any, fallback: str = "Unknown error") -> str:
        if not isinstance(payload, dict):
            return str(payload) if payload else fallback

        error = payload.get("error", payload)
        if not isinstance(error, dict):
            return str(error) if error else fallback

        parts = []
        for field in ("code", "message", "detail"):
            value = error.get(field)
            if value:
                parts.append(f"{field}={value}")
        return "; ".join(parts) if parts else str(error)

    def _check_response(
        self, response: ClientResponse, response_data: Any
    ) -> dict[str, Any]:
        status_code = response.status
        if status_code < 400:
            if isinstance(response_data, dict):
                return response_data
            raise SpecStreamServerError(
                f"SpecStream Search API returned non-object payload ({status_code})"
            )

        error_message = self._extract_error_message(response_data)
        if status_code in (429, 503):
            retry_after = response.headers.get("Retry-After")
            suffix = f" (Retry-After={retry_after})" if retry_after else ""
            raise SpecStreamRateLimitError(
                f"SpecStream Search API rate limit or busy ({status_code}){suffix}: {error_message}"
            )
        if status_code in (402, 403):
            raise SpecStreamQuotaError(
                f"SpecStream Search API billing or limit error ({status_code}): {error_message}"
            )
        if status_code == 401:
            raise ValueError(f"SpecStream Search API invalid token ({error_message})")
        if status_code in (409, 422):
            raise ValueError(
                f"SpecStream Search API invalid request ({status_code}): {error_message}"
            )
        if status_code >= 500:
            raise SpecStreamServerError(
                f"SpecStream Search API server error ({status_code}): {error_message}"
            )
        raise RuntimeError(
            f"SpecStream Search API error ({status_code}): {error_message}"
        )

    async def _execute_http_request(
        self,
        query: str,
        num_links: int,
        page: int,
        speed: str,
        safesearch: bool,
        language: str | None,
    ) -> dict[str, Any]:
        """Execute a single HTTP GET request to the SpecStream Search API."""
        session = await self._ensure_session()
        async with session.get(
            f"{_BASE_URL}{_WEB_SEARCH_ENDPOINT}",
            params=self._build_request_params(
                query=query,
                num_links=num_links,
                page=page,
                speed=speed,
                safesearch=safesearch,
                language=language,
            ),
            headers=self._build_request_headers(),
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        ) as response:
            try:
                response_data: Any = await response.json(content_type=None)
            except Exception:
                response_data = await response.text()
            return self._check_response(response, response_data)

    @retry(
        exceptions=(
            SpecStreamRateLimitError,
            SpecStreamServerError,
            ClientError,
            TimeoutError,
        ),
        is_async=True,
        delay=_DELAY,
        backoff=_BACKOFF,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
    )
    async def _execute_with_retry(
        self,
        query: str,
        num_links: int,
        page: int,
        speed: str,
        safesearch: bool,
        language: str | None,
    ) -> dict[str, Any]:
        logger.debug("Executing SpecStream Search: query=%r", query)
        await self._wait_for_rate_limit()
        response_data = await self._execute_http_request(
            query=query,
            num_links=num_links,
            page=page,
            speed=speed,
            safesearch=safesearch,
            language=language,
        )
        logger.debug("SpecStream Search successful: query=%r", query)
        return response_data

    def _parse_response(self, data: dict[str, Any]) -> list[WebSearchResult]:
        payload = data.get("data")
        if not isinstance(payload, dict):
            return []

        sources = payload.get("sources")
        if not isinstance(sources, list):
            return []

        results: list[WebSearchResult] = []
        for index, item in enumerate(sources):
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url:
                continue
            title = item.get("title")
            snippet = item.get("snippet") or item.get("excerpt")
            published = item.get("published") or item.get("modified")
            results.append(
                WebSearchResult(
                    idx=index + 1,
                    title=title if isinstance(title, str) and title else "No title",
                    url=url,
                    description=snippet if isinstance(snippet, str) else "",
                    age=(
                        published
                        if isinstance(published, str) and published
                        else "Unknown"
                    ),
                )
            )
        return results

    async def close(self) -> None:
        """Close the underlying aiohttp session if open."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed aiohttp ClientSession")
