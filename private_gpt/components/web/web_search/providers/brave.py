import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp import ClientResponse
from injector import inject, singleton

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider
from private_gpt.settings.settings import Settings
from private_gpt.utils.retry import retry

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_JITTER = (1.0, 5.0)
_BASE_URL = "https://api.search.brave.com/res/v1"
_WEB_SEARCH_ENDPOINT = "/web/search"
_GAP = 0.1  # Extra gap to query
_DELAY = 1.1
_BACKOFF = 2


class RateLimitExceeded(Exception):
    pass


class QuotaConsumed(Exception):
    """Raised when Brave Search API quota is consumed/exceeded."""

    pass


@singleton
class BraveSearchProvider(BaseWebSearchProvider):
    _initialized: bool = False
    _session: aiohttp.ClientSession | None = None
    _rate_limit_lock: asyncio.Lock | None = None
    _last_request_time: float = 0.0

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = self._settings.brave.api_key
        self._rate_limit = self._settings.brave.rate_limit
        self._rate_limit_lock: asyncio.Lock = asyncio.Lock()

    async def make_query(
        self,
        query: str,
        num_links: int = 10,
        offset: int = 0,
        result_filter: str = "web",
        safesearch: bool = True,
        freshness: str | None = None,
        spellcheck: bool = True,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:

        # 1. Check params
        query, num_links, offset = self._validate_query_params(query, num_links, offset)

        # 2. Make the request
        response_data = await self._execute_with_retry(
            query,
            num_links,
            offset,
            result_filter,
            safesearch,
            freshness,
            spellcheck,
            language,
        )

        # 3. Parser response
        return await asyncio.to_thread(self._parse_response, response_data)

    async def validate(self) -> None:
        if not self._api_key or not self._api_key.strip():
            raise ValueError("Brave Search API key is not configured")

    def _validate_query_params(
        self, query: str, num_links: int, offset: int
    ) -> tuple[str, int, int]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        normalized_num_links = max(1, min(20, num_links))  # Brave allows 1-20
        if normalized_num_links != num_links:
            logger.warning(
                f"Num_links {num_links} outside valid range [1,20], clamped to {normalized_num_links}"
            )

        if offset < 0:
            raise ValueError("Offset cannot be negative")

        return query, normalized_num_links, offset

    async def _execute_http_request(
        self,
        query: str,
        num_links: int,
        offset: int,
        result_filter: str,
        safesearch: bool,
        freshness: str | None,
        spellcheck: bool,
        language: str | None,
    ) -> Any:
        """Execute HTTP request to Brave Search API.

        This method performs the actual HTTP call without retry logic,
        allowing for easier testing and mocking.
        """
        session = await self._ensure_session()
        async with session.get(
            f"{_BASE_URL}{_WEB_SEARCH_ENDPOINT}",
            params=self._build_request_params(
                query,
                num_links,
                offset,
                result_filter,
                safesearch,
                freshness,
                spellcheck,
                language,
            ),
            headers=self._build_request_headers(),
            timeout=aiohttp.ClientTimeout(total=self._settings.brave.timeout),
        ) as response:
            response_data = await response.json()

            await asyncio.to_thread(self._check_response, response, response_data)
            return response_data

    @retry(
        exceptions=(RateLimitExceeded, Exception),
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
        offset: int,
        result_filter: str,
        safesearch: bool,
        freshness: str | None,
        spellcheck: bool,
        language: str | None,
    ) -> Any:
        """Execute Brave Search with retry logic for transient errors.

        This method wraps _execute_http_request with rate limiting and
        automatic retry on RateLimitExceeded and server errors.
        """
        logger.debug(f"Executing Brave Search: query='{query}'")
        await self._wait_for_rate_limit()

        response_data = await self._execute_http_request(
            query,
            num_links,
            offset,
            result_filter,
            safesearch,
            freshness,
            spellcheck,
            language,
        )

        logger.debug(f"Brave Search successful: query='{query}'")
        return response_data

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            enabled_proxy = self._settings.server.network.proxy.enabled
            self._session = aiohttp.ClientSession(trust_env=enabled_proxy)
        return self._session

    async def _wait_for_rate_limit(self) -> None:
        if self._rate_limit_lock is None:
            raise RuntimeError("Rate limit lock not initialized")

        async with self._rate_limit_lock:
            min_interval = (1.0 + _GAP) / self._rate_limit
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time

            wait_time = max(0, min_interval - time_since_last)

            self._last_request_time = max(now, self._last_request_time) + min_interval

            logger.debug(
                f"Rate limit check: time_since_last={time_since_last:.3f}s, "
                f"will wait {wait_time:.3f}s, next_slot={self._last_request_time:.3f}"
            )
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    def _extract_error_message(self, error: Any) -> str:
        if isinstance(error, dict):
            # Coge preferentemente 'detail', si no 'code', si no repr(dict)
            detail = error.get("detail")
            code = error.get("code")
            status = error.get("status")
            parts = []
            if code:
                parts.append(f"code={code}")
            if status:
                parts.append(f"status={status}")
            if detail:
                parts.append(f"detail={detail}")
            if parts:
                return "; ".join(parts)
            return str(error)
        return str(error)

    def _check_response(
        self, response: ClientResponse, response_data: dict[str, Any]
    ) -> None:
        quota_header = response.headers.get("x-ratelimit-remaining")
        if quota_header is not None and quota_header.strip() == "0":
            raise QuotaConsumed(
                "Brave Search API quota exhausted (x-ratelimit-remaining=0)"
            )

        status_code = response.status
        if status_code < 400:
            return

        error = response_data.get("error", "Unknown error")
        error_message = self._extract_error_message(error)
        if status_code == 429:
            logger.debug(f"Brave Search API rate limit exceeded: {error_message}")
            raise RateLimitExceeded(
                f"Brave Search API rate limit exceeded: {error_message}"
            )
        elif status_code == 400:
            raise ValueError(f"Brave Search API invalid token ({error_message})")
        elif status_code >= 500:
            raise Exception(f"Brave Search API server error: {error_message}")
        else:
            raise Exception(f"Brave Search API error ({status_code}): {error_message}")

    def _build_request_params(
        self,
        query: str,
        num_links: int,
        offset: int,
        result_filter: str,
        safesearch: bool,
        freshness: str | None,
        spellcheck: bool,
        language: str | None,
    ) -> dict[str, Any]:
        params = {
            "q": query.strip(),
            "count": num_links,
            "offset": offset,
            "result_filter": result_filter,
            "safesearch": "strict" if safesearch else "off",
            "spellcheck": "true" if spellcheck else "false",
        }
        if freshness:
            params["freshness"] = freshness
        if language:
            params["language"] = language

        return params

    def _build_request_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }

    def _parse_response(self, data: dict[str, Any]) -> list[WebSearchResult]:
        results = []
        web_results = data.get("web", {}).get("results", [])

        for index, item in enumerate(web_results):
            try:
                url = item.get("url")
                if url:
                    results.append(
                        WebSearchResult(
                            idx=index + 1,
                            title=item.get("title") or "No title",
                            url=url,
                            description=item.get("description", ""),
                            age=item.get("page_age") or item.get("age") or "Unknown",
                        )
                    )
            except Exception as e:
                logger.warning(f"Error parsing result: {e}")
                continue

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed aiohttp ClientSession")
