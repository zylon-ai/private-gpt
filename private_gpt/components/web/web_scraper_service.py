import asyncio
import importlib
import logging
import os
import re
from typing import Any

import html2text
from injector import inject, singleton
from pydantic import BaseModel

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
    _initialized: bool = False

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
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
        """Scrape a web page and return raw HTML content.

        Args:
            url: The URL to scrape

        Returns:
            The raw HTML content of the page

        Raises:
            Exception: If scraping fails
        """
        if not self._initialized:
            raise ValueError(
                "Web fetching is not properly initialized or it is disabled. "
                "Consider enabling web fetching in settings."
            )

        timeout = self._settings.web_fetch.timeout_seconds * 1000

        try:
            from playwright.async_api import (  # type: ignore[import-not-found]
                async_playwright,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "Web scraping",
                    extras="tool-web-scraping",
                )
            ) from e

        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-accelerated-2d-canvas",
                        "--no-first-run",
                        "--no-zygote",
                        "--single-process",
                        "--disable-gpu",
                    ],
                )
            except Exception as e:
                if re.search(
                    r"playwright.*install|browser.*executable|download new browsers",
                    str(e),
                    re.IGNORECASE,
                ):
                    raise RuntimeError(
                        "Playwright browsers are not installed. "
                        "Run `playwright install` and try again."
                    ) from e
                raise

            context_options: dict[str, Any] = self._get_context()
            context = await browser.new_context(**context_options)
            page = await context.new_page()

            await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("networkidle", timeout=timeout)

            html_content: str = await page.content()

            await context.close()
            await browser.close()

            return html_content

    async def scrape(self, url: str) -> WebScraperResult:
        """Scrape a web page and convert it to markdown format.

        Args:
            url: The URL to scrape

        Returns:
            The page content converted to markdown

        Raises:
            Exception: If scraping fails or no content can be extracted
        """
        result = WebScraperResult()
        result.url = url
        # Reuse the HTML scraping logic
        result.html_content = await self._scrape_html(url)

        # Convert to markdown
        result.markdown_content = await asyncio.to_thread(
            self._html_to_markdown,
            result.html_content,
        )

        if not result.markdown_content.strip():
            logger.debug(f"Cannot extract text from the provided URL: {url}")
            raise Exception(
                "Can't extract information from the provided url, "
                "automated tools do not work on this page."
            )

        return result

    async def scrape_max_compress(self, url: str) -> WebScraperResult:
        """Scrape a web page, clean HTML and convert to markdown using html2text.

        Args:
            url: The URL to scrape

        Returns:
            The cleaned and converted markdown content
        """
        result: WebScraperResult = WebScraperResult()
        result.url = url

        # 1. Scrape raw HTML
        result.html_content = await self._scrape_html(url)

        result.favicon_url = await asyncio.to_thread(
            self.get_favicon_url, url, result.html_content
        )

        # 2. Clean HTML with lxml-html-clean and convert max clean markdown
        result.markdown_content = await asyncio.to_thread(
            self._clean_html_max, result.html_content
        )
        return result

    def _get_context(self) -> dict[str, Any]:
        """Get Playwright browser context options based on settings.

        Returns:
            A dictionary of context options
        """
        context_options: dict[str, Any] = {}

        # Configure the proxy settings
        proxy_settings = self._get_proxy_settings("")
        if proxy_settings:
            context_options["proxy"] = proxy_settings

        # Configure SSL settings
        ssl_context = self._get_ssl_context()
        context_options.update(ssl_context)

        return context_options

    def _get_proxy_settings(self, url: str) -> Any | None:
        proxy_config = self._settings.server.network.proxy
        if not proxy_config.enabled:
            return None

        proxy_url = (
            proxy_config.https_server
            if url.startswith("https://")
            else proxy_config.http_server
        )
        if not proxy_url:
            logger.debug(
                f"No proxy configured for {'HTTPS' if url.startswith('https://') else 'HTTP'}"
            )
            return None

        from playwright.async_api import ProxySettings

        proxy_settings = ProxySettings(
            server=f"{proxy_url.scheme}://{proxy_url.host}"
            + (f":{proxy_url.port}" if proxy_url.port else ""),
            username=proxy_url.username,
            password=proxy_url.password,
            bypass=proxy_config.bypass or None,
        )

        logger.debug(
            f"Using proxy server: {proxy_url.scheme}://{proxy_url.host}"
            + (f":{proxy_url.port}" if proxy_url.port else "")
        )

        return proxy_settings

    def _get_ssl_context(self) -> dict[str, Any]:
        """Get SSL context options based on settings.

        Returns:
            A dictionary of SSL context options
        """
        ssl_config = self._settings.server.network.ssl
        ssl_context: dict[str, Any] = {}

        needs_ignore_errors = (
            # If SSL verification is disabled
            not ssl_config.verify_ssl
            # Since the playwright does not support custom certs directly,
            # we set ignore_https_errors to true if a custom cert is provided.
            # https://github.com/microsoft/playwright/issues/33596#issuecomment-2475461864
            or ssl_config.cert_file
        )

        if needs_ignore_errors:
            ssl_context["ignore_https_errors"] = True
            if not ssl_config.verify_ssl:
                logger.warning("SSL certificate verification is disabled.")

        return ssl_context

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML content to markdown format.

        Args:
            html: The HTML content to convert

        Returns:
            The content converted to markdown
        """
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

            # return absolute URL
            if href.startswith("http://") or href.startswith("https://"):
                return href
            from urllib.parse import urljoin

            return urljoin(page_url, href)
        except Exception:
            return None

    def _clean_html_max(self, html: str) -> str:
        """Clean HTML content deeply using lxml-html-clean, then convert to markdown."""
        try:
            # First, clean HTML with lxml_html_clean Cleaner
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

            # Then process with html2text using advanced config
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
