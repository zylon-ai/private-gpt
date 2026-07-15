import asyncio
import importlib
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import html2text
from injector import inject
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


_next_browser_id = 0


class _BrowserInstance:
    def __init__(
        self,
        browser: Any,
        context: Any,
        max_pages: int,
    ) -> None:
        global _next_browser_id
        _next_browser_id += 1
        self.id = _next_browser_id
        self.browser = browser
        self.context = context
        self.max_pages = max_pages
        self.last_used = time.monotonic()
        self.in_use_count = 0
        logger.debug(f"[Browser-{self.id}] Instance created (max_pages={max_pages})")

    @property
    def is_idle(self) -> bool:
        return self.in_use_count == 0

    @property
    def has_capacity(self) -> bool:
        return self.in_use_count < self.max_pages

    async def create_page(self) -> Any:
        self.in_use_count += 1
        self.last_used = time.monotonic()
        page = await self.context.new_page()
        logger.debug(
            f"[Browser-{self.id}] Page created (in_use={self.in_use_count}/{self.max_pages})"
        )
        return page

    async def close_page(self, page: Any) -> None:
        self.in_use_count -= 1
        self.last_used = time.monotonic()
        await page.close()
        logger.debug(
            f"[Browser-{self.id}] Page closed (in_use={self.in_use_count}/{self.max_pages})"
        )

    async def close(self) -> None:
        logger.debug(f"[Browser-{self.id}] Closing browser instance")
        await self.context.close()
        await self.browser.close()
        logger.debug(f"[Browser-{self.id}] Browser instance closed")


class _BrowserPool:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._max_pages = (
            settings.web_fetch.pool_size * settings.web_fetch.pool_max_pages_per_browser
        )
        self._sem = asyncio.Semaphore(self._max_pages)
        self._lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser: Any = None
        self._active = 0
        self._last_used = time.monotonic()
        self._closed = False

    def _available_capacity(self) -> int:
        """Pages that can be served right now without anyone having to wait."""
        return self._max_pages - self._active

    async def _ensure_browser(self) -> Any:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Browser pool is closed")
            # Relaunch if there's no browser OR the existing one died
            if self._browser is None or not self._browser.is_connected():
                if self._browser is not None:
                    logger.warning("Browser was disconnected, relaunching")
                from playwright.async_api import (  # type: ignore[import-not-found]
                    async_playwright,
                )

                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--no-first-run",
                        # "--no-zygote",
                        # "--single-process",
                    ],
                )
                logger.debug("Chromium launched")
            return self._browser

    @asynccontextmanager
    async def page(self) -> AsyncIterator[Any]:
        full = self._sem.locked()
        if full:
            logger.debug(
                f"Pool FULL, request waiting for a free slot "
                f"(active={self._active}/{self._max_pages})"
            )
        async with self._sem:  # bloquea aquí si está lleno
            async with self._lock:
                self._active += 1
                if full:
                    logger.debug(
                        f"Slot freed, page acquired after waiting "
                        f"(active={self._active}/{self._max_pages})"
                    )
                else:
                    logger.debug(
                        f"Page requested "
                        f"(active={self._active}/{self._max_pages}, "
                        f"available_capacity={self._available_capacity()})"
                    )
            try:
                browser = await self._ensure_browser()
                context = await browser.new_context(**self._context_options())
                page = await context.new_page()
                logger.debug(f"Page ACQUIRED (active={self._active}/{self._max_pages})")
                try:
                    yield page
                finally:
                    await context.close()
            finally:
                async with self._lock:
                    self._active -= 1
                    self._last_used = time.monotonic()
                    logger.debug(
                        f"Page RELEASED (active={self._active}/{self._max_pages}, "
                        f"available_capacity={self._available_capacity()})"
                    )

    def _context_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        proxy = self._proxy_settings()
        if proxy:
            opts["proxy"] = proxy
        ssl = self._settings.server.network.ssl
        if not ssl.verify_ssl or ssl.cert_file:
            opts["ignore_https_errors"] = True
            if not ssl.verify_ssl:
                logger.warning("SSL verification disabled")
        return opts

    def _proxy_settings(self) -> Any | None:
        cfg = self._settings.server.network.proxy
        if not cfg.enabled:
            return None
        url = cfg.https_server or cfg.http_server  # decided once, no dead code
        if not url:
            return None
        from playwright.async_api import ProxySettings

        return ProxySettings(
            server=f"{url.scheme}://{url.host}" + (f":{url.port}" if url.port else ""),
            username=url.username,
            password=url.password,
            bypass=cfg.bypass or None,
        )

    async def cleanup_idle(self) -> None:
        timeout = self._settings.web_fetch.pool_idle_timeout_seconds
        while not self._closed:
            await asyncio.sleep(60)
            async with self._lock:
                idle = self._browser is not None and self._active == 0
                if idle and time.monotonic() - self._last_used >= timeout:
                    browser = self._browser
                    playwright = self._playwright
                    if browser is None or playwright is None:
                        continue
                    await browser.close()
                    await playwright.stop()
                    self._browser = self._playwright = None
                    logger.debug(
                        f"Idle browser closed after {timeout}s with no activity"
                    )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            logger.debug(
                f"Shutting down browser pool "
                f"(active_pages={self._active}, "
                f"browser_alive={self._browser is not None})"
            )
            if self._browser is not None:
                await self._browser.close()
            if self._playwright is not None:
                await self._playwright.stop()
            self._browser = self._playwright = None
            logger.info("Browser pool shut down")


class WebScraperService:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: _BrowserPool | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
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

    def _ensure_pool(self) -> _BrowserPool:
        if self._pool is None:
            logger.info("Creating browser pool (first scrape request)")
            self._pool = _BrowserPool(self._settings)
            self._cleanup_task = asyncio.create_task(self._pool.cleanup_idle())
        return self._pool

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

        pool = self._ensure_pool()
        _start_acquire = time.monotonic()
        async with pool.page() as page:
            _acquire_time = time.monotonic() - _start_acquire
            logger.debug(f"Scraping {url} (acquire_time={_acquire_time:.3f}s)")
            _start_scrape = time.monotonic()
            try:
                async with asyncio.timeout(self._settings.web_fetch.timeout_seconds):
                    await page.goto(url)
                    await page.wait_for_load_state("domcontentloaded")
                    content = await page.content()
                    _scrape_time = time.monotonic() - _start_scrape
                    logger.debug(
                        f"Scraped {url} "
                        f"(content_size={len(content)}, scrape_time={_scrape_time:.3f}s)"
                    )
                    return str(content)
            except TimeoutError:
                logger.warning(
                    f"Timeout ({self._settings.web_fetch.timeout_seconds}s) scraping {url}"
                )
                raise
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
                logger.warning(f"Error scraping {url}: {e}")
                raise

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
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.error(
                    "WebScraperService cleanup task was cancelled during shutdown"
                )
                pass
            self._cleanup_task = None
        if self._pool is not None:
            logger.debug("Closing WebScraperService browser pool")
            await self._pool.close()
            self._pool = None
            logger.debug("WebScraperService closed")
        else:
            logger.debug("WebScraperService close: no pool to close")
