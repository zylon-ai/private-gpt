from unittest.mock import MagicMock

import pytest

from private_gpt.components.web.scraper import registry as scraper_registry
from private_gpt.components.web.scraper.base import WebScraperProvider
from private_gpt.components.web.scraper.local import LocalWebScraperProvider
from private_gpt.components.web.scraper.registry import (
    WebScraperProviderRegistry,
    register_web_scraper_provider,
)
from private_gpt.settings.settings import Settings


def _settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.web_fetch = MagicMock()
    settings.web_fetch.pool_size = 2
    settings.web_fetch.max_requests_per_session = 1
    settings.web_fetch.pool_idle_timeout_seconds = 300
    settings.web_fetch.batch_size = 1
    settings.web_fetch.batch_wait_ms = 0
    settings.bash = MagicMock()
    settings.bash.output_cap_bytes = 1024 * 1024
    return settings


class _FakeProvider(WebScraperProvider):
    async def scrape_html(self, url: str, timeout_seconds: int) -> str:
        return f"<html>{url}</html>"


@pytest.fixture(autouse=True)
def _restore_registry():
    original = dict(scraper_registry._PROVIDERS)
    yield
    scraper_registry._PROVIDERS.clear()
    scraper_registry._PROVIDERS.update(original)


def test_local_provider_is_the_default() -> None:
    registry = WebScraperProviderRegistry(_settings())
    provider = registry.get_provider("local")
    assert isinstance(provider, LocalWebScraperProvider)


def test_provider_instances_are_cached() -> None:
    registry = WebScraperProviderRegistry(_settings())
    assert registry.get_provider("local") is registry.get_provider("local")


def test_register_custom_provider() -> None:
    register_web_scraper_provider("fake", _FakeProvider)
    registry = WebScraperProviderRegistry(_settings())
    assert isinstance(registry.get_provider("fake"), _FakeProvider)


def test_unknown_provider_raises_with_available_names() -> None:
    registry = WebScraperProviderRegistry(_settings())
    with pytest.raises(ValueError, match=r"not registered.*local"):
        registry.get_provider("does-not-exist")


async def test_close_all_closes_instantiated_providers() -> None:
    closed: list[str] = []

    class _ClosingProvider(_FakeProvider):
        async def close(self) -> None:
            closed.append("closed")

    register_web_scraper_provider("closing", _ClosingProvider)
    registry = WebScraperProviderRegistry(_settings())
    registry.get_provider("closing")
    await registry.close_all()
    assert closed == ["closed"]
