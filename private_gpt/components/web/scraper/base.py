from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable

from private_gpt.settings.settings import Settings


class WebScraperProvider(ABC):
    """Fetches rendered page HTML by executing the shared Playwright scrape script.

    Implementations only decide WHERE the script runs (host, remote sandbox);
    the scraping logic itself is shared. Timeouts surface as ``TimeoutError``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def scrape_html(self, url: str, timeout_seconds: int) -> str:
        """Fetch the fully rendered HTML for a single URL."""

    async def scrape_many(
        self, urls: list[str], timeout_seconds: int
    ) -> list[str | BaseException]:
        """Fetch several URLs; failures are returned in-place as exceptions."""
        return await asyncio.gather(
            *(self.scrape_html(url, timeout_seconds) for url in urls),
            return_exceptions=True,
        )

    async def close(self) -> None:  # noqa: B027
        """Release provider resources. Default: no-op."""


WebScraperProviderFactory = (
    type[WebScraperProvider] | Callable[[Settings], WebScraperProvider]
)
