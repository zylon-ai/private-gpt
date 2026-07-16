from unittest.mock import MagicMock

import pytest

from private_gpt.components.web.scraper import registry as scraper_registry
from private_gpt.components.web.scraper.base import WebScraperProvider
from private_gpt.components.web.scraper.registry import register_web_scraper_provider
from private_gpt.components.web.web_scraper_service import WebScraperService
from private_gpt.settings.settings import Settings


def _settings(*, enabled: bool = True, provider: str = "fake") -> Settings:
    settings = MagicMock(spec=Settings)
    settings.web_fetch = MagicMock()
    settings.web_fetch.enabled = enabled
    settings.web_fetch.provider = provider
    settings.web_fetch.timeout_seconds = 10
    ssl = MagicMock()
    ssl.cert_file = None
    ssl.cert_dir = None
    settings.server = MagicMock()
    settings.server.network.ssl = ssl
    return settings


_PAGE_HTML = (
    "<html><head>"
    '<link rel="icon" href="/favicon.ico">'
    "</head><body><h1>Title</h1><p>Some content</p></body></html>"
)


class _FakeProvider(WebScraperProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.scraped: list[str] = []

    async def scrape_html(self, url: str, timeout_seconds: int) -> str:
        self.scraped.append(url)
        return _PAGE_HTML


@pytest.fixture(autouse=True)
def _restore_registry():
    original = dict(scraper_registry._PROVIDERS)
    yield
    scraper_registry._PROVIDERS.clear()
    scraper_registry._PROVIDERS.update(original)


async def test_scrape_delegates_to_configured_provider() -> None:
    register_web_scraper_provider("fake", _FakeProvider)
    service = WebScraperService(settings=_settings())

    result = await service.scrape("https://example.com")

    assert result.url == "https://example.com"
    assert result.html_content == _PAGE_HTML
    assert result.markdown_content is not None
    assert "Some content" in result.markdown_content


async def test_scrape_max_compress_extracts_favicon_and_markdown() -> None:
    register_web_scraper_provider("fake", _FakeProvider)
    service = WebScraperService(settings=_settings())

    result = await service.scrape_max_compress("https://example.com/page")

    assert result.favicon_url == "https://example.com/favicon.ico"
    assert result.markdown_content is not None
    assert "Some content" in result.markdown_content


async def test_disabled_web_fetch_raises() -> None:
    service = WebScraperService(settings=_settings(enabled=False))
    assert not service.is_initialized
    with pytest.raises(ValueError, match="disabled"):
        await service.scrape("https://example.com")


async def test_empty_markdown_raises() -> None:
    class _EmptyProvider(_FakeProvider):
        async def scrape_html(self, url: str, timeout_seconds: int) -> str:
            return "<html><body><script>1</script></body></html>"

    register_web_scraper_provider("fake", _EmptyProvider)
    service = WebScraperService(settings=_settings())

    with pytest.raises(Exception, match="Can't extract information"):
        await service.scrape("https://example.com")


def test_get_favicon_url_handles_absolute_urls() -> None:
    service = WebScraperService(settings=_settings(enabled=False))
    html = '<link rel="shortcut icon" href="https://cdn.example.com/i.png">'
    assert (
        service.get_favicon_url("https://example.com", html)
        == "https://cdn.example.com/i.png"
    )
    assert service.get_favicon_url("https://example.com", "<html></html>") is None
