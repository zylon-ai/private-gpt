import asyncio
import importlib
import logging
import os
import re
import time
from typing import Any

import html2text
from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.web.scraper.registry import WebScraperProviderRegistry
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

logger = logging.getLogger(__name__)


def _get_html_cleaner() -> Any:
    try:
        return importlib.import_module("lxml_html_clean").Cleaner
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Web fetch markdown cleaning",
                extras="ingest-markup",
            )
        ) from e


class WebScraperResult(BaseModel):
    url: str | None = None
    html_content: str | None = None
    markdown_content: str | None = None
    favicon_url: str | None = None


@singleton
class WebScraperService:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = WebScraperProviderRegistry(settings)
        self._initialized: bool = False
        self._initialize()

    def _initialize(self) -> None:
        if self._initialized:
            logger.debug("WebScraperService already initialized, skipping")
            return
        if not self._settings.web_fetch.enabled:
            logger.warning(
                "Web fetching is disabled in settings. Skipping initialization."
            )
            return
        self._apply_ssl_environment()
        self._initialized = True
        logger.debug("WebScraperService initialized successfully")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _apply_ssl_environment(self) -> None:
        ssl_config = self._settings.server.network.ssl
        if ssl_config.cert_file:
            os.environ["NODE_EXTRA_CA_CERTS"] = ssl_config.cert_file
            os.environ["SSL_CERT_FILE"] = ssl_config.cert_file
            logger.debug(f"Custom CA certificate configured: {ssl_config.cert_file}")

        if ssl_config.cert_dir:
            os.environ["SSL_CERT_DIR"] = ssl_config.cert_dir
            logger.debug(
                f"Custom CA certificate directory configured: {ssl_config.cert_dir}"
            )

    async def _scrape_html(self, url: str) -> str:
        if not self._initialized:
            raise ValueError(
                "Web fetching is not properly initialized or it is disabled. "
                "Consider enabling web fetching in settings."
            )

        provider = self._registry.get_provider(self._settings.web_fetch.provider)
        _start_scrape = time.monotonic()
        try:
            content = await provider.scrape_html(
                url, self._settings.web_fetch.timeout_seconds
            )
        except TimeoutError:
            logger.warning(
                f"Timeout ({self._settings.web_fetch.timeout_seconds}s) scraping {url}"
            )
            raise
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            raise
        _scrape_time = time.monotonic() - _start_scrape
        logger.debug(
            f"Scraped {url} "
            f"(content_size={len(content)}, scrape_time={_scrape_time:.3f}s)"
        )
        return content

    async def scrape(self, url: str) -> WebScraperResult:
        _start = time.monotonic()
        logger.debug(f"Scrape start: {url}")
        result = WebScraperResult()
        result.url = url
        result.html_content = await self._scrape_html(url)

        result.markdown_content = await asyncio.to_thread(
            self._html_to_markdown,
            result.html_content,
        )

        if not result.markdown_content.strip():
            _elapsed = time.monotonic() - _start
            logger.warning(f"Cannot extract text from {url} ({_elapsed:.2f}s)")
            raise Exception(
                "Can't extract information from the provided url, "
                "automated tools do not work on this page."
            )
        _elapsed = time.monotonic() - _start
        logger.debug(
            f"Scrape complete: {url} ({_elapsed:.2f}s, "
            f"html={len(result.html_content or '')}, "
            f"md={len(result.markdown_content or '')})"
        )
        return result

    async def scrape_max_compress(self, url: str) -> WebScraperResult:
        _start = time.monotonic()
        logger.debug(f"Max-compress scrape start: {url}")
        result: WebScraperResult = WebScraperResult()
        result.url = url

        result.html_content = await self._scrape_html(url)

        result.favicon_url = await asyncio.to_thread(
            self.get_favicon_url, url, result.html_content
        )

        result.markdown_content = await asyncio.to_thread(
            self._clean_html_max, result.html_content
        )

        _elapsed = time.monotonic() - _start
        logger.debug(
            f"Max-compress scrape complete: {url} ({_elapsed:.2f}s, "
            f"html={len(result.html_content or '')}, "
            f"md={len(result.markdown_content or '')})"
        )
        return result

    def _html_to_markdown(self, html: str) -> str:
        h: html2text.HTML2Text = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_tables = True
        h.body_width = 0
        h.single_line_break = True

        result: str = h.handle(html).strip()
        return result

    def get_favicon_url(self, page_url: str, html: str) -> str | None:
        try:
            m = re.search(
                r'<link[^>]*rel=["\']?[^"\'>]*icon[^"\'>]*["\']?[^>]*>',
                html,
                flags=re.IGNORECASE,
            )
            if not m:
                return None

            link_tag = m.group(0)

            m_href = re.search(
                r'href=["\']([^"\']+)["\']', link_tag, flags=re.IGNORECASE
            )
            if not m_href:
                return None

            href = m_href.group(1)

            if href.startswith("http://") or href.startswith("https://"):
                return href
            from urllib.parse import urljoin

            return urljoin(page_url, href)
        except Exception:
            return None

    def _clean_html_max(self, html: str) -> str:
        try:
            cleaner_cls = _get_html_cleaner()
            cleaner = cleaner_cls(
                scripts=True,
                javascript=True,
                comments=True,
                style=True,
                inline_style=True,
                links=True,
                meta=True,
                page_structure=True,
                processing_instructions=True,
                embedded=True,
                frames=True,
                forms=True,
                annoying_tags=True,
                remove_unknown_tags=True,
                safe_attrs_only=True,
                add_nofollow=True,
            )
            cleaned_html = cleaner.clean_html(html)

            h = html2text.HTML2Text()

            h.ignore_links = True
            h.ignore_images = True
            h.ignore_tables = True
            h.ignore_emphasis = True
            h.single_line_break = True
            h.body_width = 0
            h.protect_links = False
            h.unicode_snob = True
            h.bypass_tables = True

            markdown = h.handle(cleaned_html).strip()

            return markdown

        except Exception:
            return html

    async def close(self) -> None:
        await self._registry.close_all()
        logger.debug("WebScraperService closed")
