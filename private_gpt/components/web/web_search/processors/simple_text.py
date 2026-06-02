import logging

from injector import inject

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.base import (
    BaseWebSearchResultProcessor,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class SimpleTextProcessor(BaseWebSearchResultProcessor):
    @inject
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        super().__init__()
        self._settings = settings

    async def process_results(
        self,
        query: str,
        results: list[WebSearchResult],
        model_id: str | None = None,
    ) -> list[WebSearchResult]:
        return results
