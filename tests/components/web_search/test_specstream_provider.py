from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.components.web.web_search.providers.specstream import (
    SpecStreamQuotaError,
    SpecStreamRateLimitError,
    SpecStreamSearchProvider,
    SpecStreamServerError,
)
from private_gpt.settings.settings import Settings


@pytest.fixture
def mock_settings_factory():
    """Factory to create minimal Settings for SpecStreamSearchProvider."""

    def _create_settings(
        api_key: str = "specstream-test-key",
        speed: str = "fast",
        rate_limit: float = 0.0,
        timeout: int = 30,
        proxy_enabled: bool = False,
    ) -> Settings:
        settings = MagicMock(spec=Settings)
        settings.specstream = MagicMock()
        settings.specstream.api_key = api_key
        settings.specstream.speed = speed
        settings.specstream.rate_limit = rate_limit
        settings.specstream.timeout = timeout
        settings.server = MagicMock()
        settings.server.network = MagicMock()
        settings.server.network.proxy = MagicMock()
        settings.server.network.proxy.enabled = proxy_enabled
        return settings

    return _create_settings


@pytest.fixture
def mock_specstream_json_response() -> dict[str, Any]:
    """Mock JSON response from SpecStream Search API."""
    return {
        "data": {
            "kind": "fact",
            "query": "python asyncio",
            "sources": [
                {
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "title": "asyncio - Asynchronous I/O",
                    "site": "docs.python.org",
                    "snippet": "asyncio is a library to write concurrent code.",
                    "published": "2026-01-15",
                    "cited": True,
                },
                {
                    "url": "https://example.com/no-published",
                    "title": "",
                    "site": "example.com",
                    "excerpt": "Excerpt fallback text.",
                    "modified": "2026-02-01",
                    "cited": False,
                },
                {
                    "title": "Missing URL entry",
                    "snippet": "Should be skipped.",
                },
            ],
            "took_ms": 420,
            "price_usd": 0.001,
        }
    }


def _mock_http_session(
    status: int,
    json_payload: Any = None,
    text_payload: str = "",
    headers: dict[str, str] | None = None,
    json_side_effect: Exception | None = None,
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.headers = headers or {}
    if json_side_effect is not None:
        mock_response.json = AsyncMock(side_effect=json_side_effect)
    else:
        mock_response.json = AsyncMock(return_value=json_payload)
    mock_response.text = AsyncMock(return_value=text_payload)

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.get.return_value = mock_context
    return mock_session


class TestSpecStreamProviderValidation:
    """Tests for SpecStreamSearchProvider configuration and parameter validation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("api_key", ["", "   "])
    async def test_validate_missing_api_key_raises(
        self, mock_settings_factory, api_key
    ):
        provider = SpecStreamSearchProvider(mock_settings_factory(api_key=api_key))
        with pytest.raises(
            ValueError, match="SpecStream Search API key is not configured"
        ):
            await provider.validate()

    @pytest.mark.asyncio
    async def test_validate_configured_api_key_passes(self, mock_settings_factory):
        provider = SpecStreamSearchProvider(mock_settings_factory(api_key="valid-key"))
        await provider.validate()

    def test_validate_query_params_checks_bounds(self, mock_settings_factory):
        provider = SpecStreamSearchProvider(mock_settings_factory())

        with pytest.raises(ValueError, match="Search query cannot be empty"):
            provider._validate_query_params("   ", 10, 0)

        with pytest.raises(ValueError, match="cannot exceed 2000 characters"):
            provider._validate_query_params("x" * 2001, 10, 0)

        with pytest.raises(ValueError, match="Offset cannot be negative"):
            provider._validate_query_params("valid", 10, -1)

        with pytest.raises(ValueError, match="Unsupported SpecStream speed"):
            provider._validate_query_params("valid", 10, 0, speed="turbo")

        q, num, page, speed = provider._validate_query_params(
            "  hello world  ", 250, 20, speed="default"
        )
        assert (q, num, page, speed) == ("hello world", 100, 1, "default")

        _, num2, page2, _ = provider._validate_query_params("hello", 10, 20)
        assert (num2, page2) == (10, 3)


class TestSpecStreamProviderRequests:
    """Tests for request construction, response parsing, and error handling."""

    @pytest.mark.asyncio
    async def test_make_query_success_maps_sources(
        self, mock_settings_factory, mock_specstream_json_response
    ):
        provider = SpecStreamSearchProvider(
            mock_settings_factory(api_key="test-key", speed="fast")
        )
        mock_session = _mock_http_session(
            200, json_payload=mock_specstream_json_response
        )

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            results = await provider.make_query(
                "  python asyncio  ",
                num_links=10,
                offset=10,
                safesearch=False,
                language="en",
            )

        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert call_args.args[0] == "https://api.specstream.ai/v1/web-search"
        assert call_args.kwargs["params"] == {
            "q": "python asyncio",
            "num": 10,
            "speed": "fast",
            "safe": "off",
            "page": 2,
            "language": "en",
        }
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"

        assert len(results) == 2
        assert results[0].idx == 1
        assert results[0].title == "asyncio - Asynchronous I/O"
        assert results[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert (
            results[0].description == "asyncio is a library to write concurrent code."
        )
        assert results[0].age == "2026-01-15"

        assert results[1].idx == 2
        assert results[1].title == "No title"
        assert results[1].url == "https://example.com/no-published"
        assert results[1].description == "Excerpt fallback text."
        assert results[1].age == "2026-02-01"

    @pytest.mark.asyncio
    async def test_make_query_empty_sources_returns_empty_list(
        self, mock_settings_factory
    ):
        provider = SpecStreamSearchProvider(mock_settings_factory())
        mock_session = _mock_http_session(200, json_payload={"data": {"sources": []}})

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            results = await provider.make_query("site:nonexistent.invalid")

        assert results == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "payload", "exc_type", "match"),
        [
            (
                401,
                {"error": {"code": "invalid_token", "message": "Invalid API key"}},
                ValueError,
                "invalid token",
            ),
            (
                422,
                {"error": {"code": "validation_error", "message": "q is too long"}},
                ValueError,
                "invalid request",
            ),
            (
                402,
                {"error": {"code": "insufficient_balance", "message": "Add funds"}},
                SpecStreamQuotaError,
                "billing or limit error",
            ),
            (
                403,
                {"error": {"code": "forbidden", "message": "Spend cap reached"}},
                SpecStreamQuotaError,
                "billing or limit error",
            ),
            (
                429,
                {"error": {"code": "rate_limited", "message": "Too many requests"}},
                SpecStreamRateLimitError,
                "Retry-After=5",
            ),
            (
                503,
                {"error": {"code": "busy", "message": "Search busy"}},
                SpecStreamRateLimitError,
                "rate limit or busy",
            ),
            (
                500,
                {"error": {"code": "internal_error", "message": "Server failure"}},
                SpecStreamServerError,
                "server error",
            ),
        ],
    )
    async def test_http_error_mapping(
        self, mock_settings_factory, status, payload, exc_type, match
    ):
        provider = SpecStreamSearchProvider(mock_settings_factory())
        headers = {"Retry-After": "5"} if status == 429 else {}
        mock_session = _mock_http_session(status, json_payload=payload, headers=headers)

        with (
            patch.object(provider, "_ensure_session", return_value=mock_session),
            pytest.raises(exc_type, match=match),
        ):
            await provider._execute_http_request(
                query="test",
                num_links=10,
                page=1,
                speed="fast",
                safesearch=True,
                language=None,
            )

    @pytest.mark.asyncio
    async def test_non_json_error_response_falls_back_to_text(
        self, mock_settings_factory
    ):
        provider = SpecStreamSearchProvider(mock_settings_factory())
        mock_session = _mock_http_session(
            502,
            text_payload="Bad Gateway",
            json_side_effect=ValueError("invalid json"),
        )

        with (
            patch.object(provider, "_ensure_session", return_value=mock_session),
            pytest.raises(SpecStreamServerError, match="Bad Gateway"),
        ):
            await provider._execute_http_request(
                query="test",
                num_links=10,
                page=1,
                speed="fast",
                safesearch=True,
                language=None,
            )

    @pytest.mark.asyncio
    async def test_non_retried_error_does_not_retry_in_make_query(
        self, mock_settings_factory
    ):
        provider = SpecStreamSearchProvider(mock_settings_factory())
        mock_session = _mock_http_session(
            401,
            json_payload={"error": {"code": "invalid_token", "message": "Bad key"}},
        )

        with (
            patch.object(provider, "_ensure_session", return_value=mock_session),
            pytest.raises(ValueError, match="invalid token"),
        ):
            await provider.make_query("test query")

        assert mock_session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_close_closes_open_session(self, mock_settings_factory):
        provider = SpecStreamSearchProvider(mock_settings_factory())
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_awaited_once()
