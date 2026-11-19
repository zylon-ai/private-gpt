import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.select_best_links import (
    SelectBestLinks,
)
from private_gpt.settings.settings import WebSearchParams


def _processor(scrape: AsyncMock) -> SelectBestLinks:
    settings = MagicMock()
    settings.web_search.mode_quality = "fast"
    settings.web_search.fast_params = WebSearchParams(
        max_timeout_seconds=1, num_references=3
    )
    settings.web_search.context_token = 2000
    settings.web_search.num_links = 10
    scraper = MagicMock()
    scraper.scrape_max_compress = scrape
    return SelectBestLinks(settings, scraper, MagicMock(), MagicMock())


def _results(count: int) -> list[WebSearchResult]:
    return [
        WebSearchResult(
            idx=i, title=f"Listing {i}", url=f"https://x/{i}", description=f"snip {i}"
        )
        for i in range(count)
    ]


@pytest.mark.anyio
async def test_falls_back_to_provider_snippets_when_no_page_can_be_scraped() -> None:
    processor = _processor(AsyncMock(side_effect=RuntimeError("blocked")))

    selected = await processor.process_results("query", _results(5))

    assert [r.url for r in selected] == ["https://x/0", "https://x/1", "https://x/2"]
    assert all(r.content is None and r.description for r in selected)


@pytest.mark.anyio
async def test_falls_back_to_provider_snippets_on_timeout() -> None:
    async def never_finishes(url: str) -> SimpleNamespace:
        await asyncio.sleep(10)
        return SimpleNamespace(markdown_content="page", favicon_url=None)

    processor = _processor(AsyncMock(side_effect=never_finishes))

    selected = await processor.process_results("query", _results(4))

    assert len(selected) == 3
    assert all(r.content is None and r.description for r in selected)
