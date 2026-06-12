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
        logger.debug(
            f"[Browser-{self.id}] Instance created (max_pages={max_pages})"
        )

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
        self._playwright: Any = None
        self._browsers: list[_BrowserInstance] = []
        self._lock = asyncio.Lock()
        self._page_available = asyncio.Condition(self._lock)
        self._cleanup_task: asyncio.Task[None] | None = None
        self._closed = False
        self._waiters_count = 0

    async def _ensure_playwright(self) -> None:
        if self._playwright is None:
            logger.info("Starting Playwright (first use)")
            try:
                from playwright.async_api import async_playwright
            except ImportError as e:
                raise ImportError(
                    format_missing_dependency_message(
                        "Web scraping",
                        extras="tool-web-scraping",
                    )
                ) from e

            self._playwright = await async_playwright().start()
            logger.info("Playwright started successfully")

    async def _launch_browser(self) -> _BrowserInstance:
        logger.debug(
            f"Launching new Chromium browser "
            f"(current={len(self._browsers)}, max={self._settings.web_fetch.pool_size})"
        )
        browser_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--no-first-run",
            "--no-zygote",
            "--single-process",
            "--disable-gpu",
        ]
        browser = await self._playwright.chromium.launch(
            headless=True,
            args=browser_args,
        )
        context_options: dict[str, Any] = {}

        proxy_settings = self._get_proxy_settings("")
        if proxy_settings:
            context_options["proxy"] = proxy_settings
            logger.debug("Proxy settings applied to browser context")

        ssl_context = self._get_ssl_context()
        context_options.update(ssl_context)

        context = await browser.new_context(**context_options)
        instance = _BrowserInstance(
            browser=browser,
            context=context,
            max_pages=self._settings.web_fetch.pool_max_pages_per_browser,
        )
        self._browsers.append(instance)
        logger.debug(
            f"[Browser-{instance.id}] Launched. Pool now has {len(self._browsers)} browser(s)"
        )
        return instance

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
            return None
        from playwright.async_api import ProxySettings

        return ProxySettings(
            server=f"{proxy_url.scheme}://{proxy_url.host}"
            + (f":{proxy_url.port}" if proxy_url.port else ""),
            username=proxy_url.username,
            password=proxy_url.password,
            bypass=proxy_config.bypass or None,
        )

    def _get_ssl_context(self) -> dict[str, Any]:
        ssl_config = self._settings.server.network.ssl
        ssl_context: dict[str, Any] = {}
        needs_ignore_errors = (
            not ssl_config.verify_ssl or ssl_config.cert_file
        )
        if needs_ignore_errors:
            ssl_context["ignore_https_errors"] = True
            if not ssl_config.verify_ssl:
                logger.warning("SSL certificate verification is disabled.")
        return ssl_context

    async def acquire_page(self) -> tuple[Any, _BrowserInstance]:
        async with self._page_available:
            if self._closed:
                raise RuntimeError("Browser pool is closed")
            await self._ensure_playwright()
            while True:
                for instance in self._browsers:
                    if instance.has_capacity:
                        page = await instance.create_page()
                        logger.debug(
                            f"[Browser-{instance.id}] Page acquired "
                            f"(in_use={instance.in_use_count}/{instance.max_pages}, "
                            f"pool_browsers={len(self._browsers)})"
                        )
                        return page, instance

                if len(self._browsers) < self._settings.web_fetch.pool_size:
                    instance = await self._launch_browser()
                    page = await instance.create_page()
                    logger.debug(
                        f"[Browser-{instance.id}] New browser page acquired "
                        f"(pool_browsers={len(self._browsers)})"
                    )
                    return page, instance

                logger.debug(
                    f"All {len(self._browsers)} browser(s) at capacity "
                    f"(max_pages={self._settings.web_fetch.pool_max_pages_per_browser}), "
                    f"waiting for release..."
                )
                self._waiters_count += 1
                await self._page_available.wait()
                self._waiters_count -= 1
                if self._closed:
                    raise RuntimeError("Browser pool is closed")

    async def release_page(self, page: Any, instance: _BrowserInstance) -> None:
        async with self._page_available:
            await instance.close_page(page)
            logger.debug(
                f"[Browser-{instance.id}] Page released "
                f"(in_use={instance.in_use_count}/{instance.max_pages}, "
                f"waiters={self._waiters_count}, pool_browsers={len(self._browsers)})"
            )
            self._page_available.notify_all()

    async def _cleanup_idle(self) -> None:
        logger.debug("Idle browser cleanup task started (cycle every 60s)")
        await asyncio.sleep(60)
        while not self._closed:
            try:
                await self._cleanup_once()
            except Exception as e:
                logger.error(f"Error during browser cleanup: {e}")
            await asyncio.sleep(60)
        logger.debug("Idle browser cleanup task stopped")

    async def _cleanup_once(self) -> None:
        idle_timeout = self._settings.web_fetch.pool_idle_timeout_seconds
        now = time.monotonic()
        async with self._page_available:
            if not self._browsers:
                logger.debug("Cleanup: no browsers in pool")
                return
            to_remove = []
            for instance in self._browsers:
                idle_for = now - instance.last_used
                if instance.is_idle and idle_for >= idle_timeout:
                    to_remove.append(instance)
            if to_remove:
                for instance in to_remove:
                    self._browsers.remove(instance)
                    await instance.close()
                if not self._browsers and self._playwright is not None:
                    await self._playwright.__aexit__(None, None, None)
                    self._playwright = None  #
                logger.info(
                    f"Cleanup closed {len(to_remove)} idle browser(s), "
                    f"pool now has {len(self._browsers)} active"
                )
            else:
                idle = sum(1 for b in self._browsers if b.is_idle)
                busy = sum(1 for b in self._browsers if not b.is_idle)
                logger.debug(
                    f"Cleanup: pool={len(self._browsers)} "
                    f"(idle={idle}, busy={busy}, timeout={idle_timeout}s)"
                )

    def start_cleanup(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            logger.debug("Starting idle browser cleanup task")
            self._cleanup_task = asyncio.create_task(self._cleanup_idle())
        else:
            logger.debug("Cleanup task already running")

    async def close_all(self) -> None:
        logger.info(
            f"Shutting down browser pool: "
            f"closing {len(self._browsers)} browser(s)"
        )
        self._closed = True
        if self._cleanup_task and not self._cleanup_task.done():
            logger.debug("Cancelling cleanup task")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        async with self._page_available:
            for instance in self._browsers:
                await instance.close()
            browser_count = len(self._browsers)
            self._browsers.clear()
            self._page_available.notify_all()
            if self._playwright is not None:
                logger.info("Stopping Playwright")
                await self._playwright.stop()
                self._playwright = None
        logger.info(f"Browser pool shut down: {browser_count} browser(s) closed")


class WebScraperService:

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: _BrowserPool | None = None
        self._initialized: bool = False
        self._initialize()

    def _initialize(self) -> None:
        if self._initialized:
            logger.debug("WebScraperService already initialized, skipping")
            return
        if not self._settings.web_fetch.enabled:
            logger.warning("Web fetching is disabled in settings. Skipping initialization.")
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
            self._pool.start_cleanup()
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
        page, instance = await pool.acquire_page()
        _acquire_time = time.monotonic() - _start_acquire
        logger.debug(
            f"[Browser-{instance.id}] Scraping {url} "
            f"(acquire_time={_acquire_time:.3f}s)"
        )
        _start_scrape = time.monotonic()
        try:
            async with asyncio.timeout(self._settings.web_fetch.timeout_seconds):
                await page.goto(url)
                await page.wait_for_load_state("domcontentloaded")
                content = await page.content()
                _scrape_time = time.monotonic() - _start_scrape
                logger.debug(
                    f"[Browser-{instance.id}] Scraped {url} "
                    f"(content_size={len(content)}, scrape_time={_scrape_time:.3f}s)"
                )
                return content
        except asyncio.TimeoutError:
            logger.warning(
                f"[Browser-{instance.id}] Timeout ({self._settings.web_fetch.timeout_seconds}s) "
                f"scraping {url}"
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
            logger.warning(
                f"[Browser-{instance.id}] Error scraping {url}: {e}"
            )
            raise
        finally:
            await pool.release_page(page, instance)

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
            logger.warning(
                f"Cannot extract text from {url} ({_elapsed:.2f}s)"
            )
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
        if self._pool is not None:
            logger.info("Closing WebScraperService browser pool")
            await self._pool.close_all()
            self._pool = None
            logger.info("WebScraperService closed")
        else:
            logger.debug("WebScraperService close: no pool to close")
