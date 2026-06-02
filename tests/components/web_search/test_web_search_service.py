import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.web_search_service import WebSearchService
from private_gpt.events.models import ResultContentBlockType, TextBlock
from private_gpt.settings.settings import Settings


@pytest.fixture
def mock_settings_factory():
    """Factory to create Settings with different configurations."""

    def _create_settings(
        enabled: bool = True,
        provider: str = "mock",
        processor: str = "best_links",
        config_brave: bool = False,
        **kwargs: Any,
    ) -> Settings:
        # Create a minimal Settings object with web_search configuration
        settings = MagicMock(spec=Settings)
        settings.web_search = MagicMock()
        settings.web_search.enabled = enabled
        settings.web_search.provider = provider
        settings.web_search.processor = processor
        if config_brave:
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
def sample_web_search_results() -> list[WebSearchResult]:
    """Sample WebSearchResult list for testing."""
    return [
        WebSearchResult(
            title="Test Result 1",
            url="https://example.com/page1",
            description="This is a test snippet for page 1",
        ),
        WebSearchResult(
            title="Test Result 2",
            url="https://example.com/page2",
            description="This is a test snippet for page 2",
        ),
        WebSearchResult(
            title="Test Result 3",
            url="https://example.com/page3",
            description="This is a test snippet for page 3",
        ),
    ]


@pytest.fixture
def mock_provider():
    """Mock BaseWebSearchProvider for testing."""
    provider = AsyncMock()
    provider.make_query = AsyncMock()
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def mock_processor():
    """Mock BaseWebSearchResultProcessor for testing."""
    processor = AsyncMock()
    processor.process_results = AsyncMock()
    return processor


@pytest.fixture
def sample_result_content_blocks() -> list[ResultContentBlockType]:
    """Sample processed result content blocks."""
    return [
        TextBlock(text="Processed content from page 1"),
        TextBlock(text="Processed content from page 2"),
    ]


class TestWebSearchServiceConfiguration:
    """Tests for WebSearchService configuration validation."""

    def test_invalid_provider_raises_value_error(self, mock_settings_factory):
        """Invalid provider should raise ValueError with specific message."""
        settings = mock_settings_factory(enabled=True, provider="invalid_provider")

        with pytest.raises(ValueError) as exc_info:
            WebSearchService(
                settings=settings,
                scraper_service=MagicMock(),
                llm_component=MagicMock(),
                summary_builder=MagicMock(),
            )

        assert "Unsupported web search provider: invalid_provider" in str(
            exc_info.value
        )

    def test_invalid_processor_raises_value_error(self, mock_settings_factory):
        """Invalid processor should raise ValueError with specific message."""
        settings = mock_settings_factory(
            enabled=True, provider="mock", processor="invalid_processor"
        )

        with pytest.raises(ValueError) as exc_info:
            WebSearchService(
                settings=settings,
                scraper_service=MagicMock(),
                llm_component=MagicMock(),
                summary_builder=MagicMock(),
            )

        assert "Unknown processor: invalid_processor" in str(exc_info.value)
        assert "Available: simple_text, scraped_content, clean_content" in str(
            exc_info.value
        )

    def test_web_search_disabled_logs_error(self, mock_settings_factory, caplog):
        """When web_search.enabled=False, should log error and not initialize."""
        caplog.set_level(logging.WARNING)
        settings = mock_settings_factory(enabled=False)

        service = WebSearchService(
            settings=settings,
            scraper_service=MagicMock(),
            llm_component=MagicMock(),
            summary_builder=MagicMock(),
        )

        assert (
            "Web Search is disabled in settings, skipping initialization" in caplog.text
        )
        assert not service._initialized


class TestWebSearchServiceInitialization:
    """Tests for WebSearchService initialization logic."""

    def test_double_initialization_skips_second(self, mock_settings_factory, caplog):
        """Calling _initialize() twice should skip the second initialization."""
        caplog.set_level(logging.DEBUG)
        settings = mock_settings_factory(enabled=True, provider="mock")

        service = WebSearchService(
            settings=settings,
            scraper_service=MagicMock(),
            llm_component=MagicMock(),
            summary_builder=MagicMock(),
        )
        assert service._initialized

        # Call initialize again
        service._initialize()

        assert "WebSearchService already initialized, skipping" in caplog.text

    @pytest.mark.parametrize(
        ("provider", "expected_class"),
        [("brave", "BraveSearchProvider")],
    )
    def test_each_provider_initializes_correctly(
        self, mock_settings_factory, provider, expected_class
    ):
        """Each provider type should initialize the correct class."""
        settings = mock_settings_factory(
            enabled=True, provider=provider, config_brave=True
        )

        settings.web_search.cached = False

        service = WebSearchService(
            settings=settings,
            scraper_service=MagicMock(),
            llm_component=MagicMock(),
            summary_builder=MagicMock(),
        )

        assert service._provider.__class__.__name__ == expected_class

    @pytest.mark.parametrize(
        ("provider", "expected_class"),
        [("brave", "CachedProvider")],
    )
    def test_each_provider_initializes_correctly_cached_true(
        self, mock_settings_factory, provider, expected_class
    ):
        """Each provider type should initialize the correct class."""
        settings = mock_settings_factory(
            enabled=True, provider=provider, config_brave=True
        )

        service = WebSearchService(
            settings=settings,
            scraper_service=MagicMock(),
            llm_component=MagicMock(),
            summary_builder=MagicMock(),
        )

        assert service._provider.__class__.__name__ == expected_class
