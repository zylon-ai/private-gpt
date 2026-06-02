from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.components.web.web_search.providers.brave import (
    BraveSearchProvider,
    QuotaConsumed,
)
from private_gpt.settings.settings import Settings


@pytest.fixture
def mock_settings_factory():
    """Factory to create Settings with different configurations."""

    def _create_settings(
        enabled: bool = True,
        provider: str = "mock",
        processor: str = "simple_text",
        **kwargs: Any,
    ) -> Settings:
        # Create a minimal Settings object with web_search configuration
        settings = MagicMock(spec=Settings)
        settings.web_search = MagicMock()
        settings.web_search.enabled = enabled
        settings.web_search.provider = provider
        settings.web_search.processor = processor
        settings.brave = MagicMock()
        settings.brave.api_key = "api-key"
        settings.brave.rate_limit = 1.0
        settings.brave.timeout = 30

        # Add any additional kwargs to settings
        for key, value in kwargs.items():
            setattr(settings.web_search, key, value)

        return settings

    return _create_settings


@pytest.fixture
def mock_brave_json_response() -> dict[str, Any]:
    """Mock JSON response from Brave API."""
    return {
        "web": {
            "results": [
                {
                    "title": "Example Domain",
                    "url": "https://example.com",
                    "description": "Example Domain. This domain is for use in illustrative examples.",
                    "age": "2023-01-15T10:30:00Z",
                },
                {
                    "title": "Test Page",
                    "url": "https://test.com/page",
                    "description": "A test page with useful information",
                    "age": "2023-02-20T14:45:00Z",
                },
            ]
        }
    }


@pytest.fixture
def mock_422_response():
    """Mock aiohttp ClientResponse for 422 error."""
    return {
        "error": {
            "code": "SUBSCRIPTION_TOKEN_INVALID",
            "detail": "The provided subscription token is invalid.",
            "meta": {"component": "authentication"},
            "status": 422,
        },
        "type": "ErrorResponse",
    }


class TestBraveProviderErrorHandling:
    """Tests for BraveSearchProvider HTTP error handling."""

    @pytest.mark.asyncio
    async def test_subscription_token_invalid_error(
        self, mock_settings_factory, mock_422_response
    ):
        """HTTP 422 with token error should raise a meaningful ValueError (no retry)."""
        settings = mock_settings_factory(enabled=True, provider="brave")
        provider = BraveSearchProvider(settings)

        # Mock the aiohttp response
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value=mock_422_response)

        # Mock the context manager for session.get()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        # Mock the session
        mock_session = MagicMock()
        mock_session.get.return_value = mock_context

        # Patch _ensure_session to return our mock session
        with patch.object(
            provider, "_ensure_session", return_value=mock_session
        ), pytest.raises(ValueError) as exc:
            await provider._execute_http_request(
                query="test query",
                num_links=10,
                offset=0,
                result_filter="web",
                safesearch=True,
                freshness=None,
                spellcheck=True,
                language=None,
            )

        # Verify error message
        msg = str(exc.value)
        assert "Brave Search API invalid token" in msg
        assert "SUBSCRIPTION_TOKEN_INVALID" in msg

    @pytest.mark.asyncio
    async def test_rate_limit_header_ok(
        self, mock_settings_factory, mock_brave_json_response
    ):
        """Test response [200 OK] with x-ratelimit-remaining > 0 (should work fine)."""
        settings = mock_settings_factory(enabled=True, provider="brave")
        provider = BraveSearchProvider(settings)

        # Mock the response object: status ok and headers with limit remaining
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_brave_json_response)
        mock_response.headers = {
            "x-ratelimit-remaining": "1",  # Enough quota
            "Content-Type": "application/json",
        }

        # Mock async context manager for session.get()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_context

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider._execute_http_request(
                query="python asyncio",
                num_links=10,
                offset=0,
                result_filter="web",
                safesearch=True,
                freshness=None,
                spellcheck=True,
                language=None,
            )
            assert isinstance(result, dict)
            assert "web" in result

    @pytest.mark.asyncio
    async def test_rate_limit_header_exceeded(
        self, mock_settings_factory, mock_brave_json_response
    ):
        settings = mock_settings_factory(enabled=True, provider="brave")
        provider = BraveSearchProvider(settings)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_brave_json_response)
        mock_response.headers = {
            "x-ratelimit-remaining": "0",  # No quota left
            "Content-Type": "application/json",
        }

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_context

        with patch.object(
            provider, "_ensure_session", return_value=mock_session
        ), pytest.raises(QuotaConsumed):
            await provider._execute_http_request(
                query="python asyncio",
                num_links=10,
                offset=0,
                result_filter="web",
                safesearch=True,
                freshness=None,
                spellcheck=True,
                language=None,
            )
