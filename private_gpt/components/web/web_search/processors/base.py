from abc import ABC, abstractmethod

from private_gpt.components.web.web_search.models import WebSearchResult


class BaseWebSearchResultProcessor(ABC):
    """Base class for web search result processors.

    This abstract class defines the interface that all web search result
    processors must implement. Processors are responsible for taking raw
    search results and transforming them into structured content blocks
    that can be used by the application.
    """

    @abstractmethod
    async def process_results(
        self,
        query: str,
        results: list[WebSearchResult],
        model_id: str | None = None,
    ) -> list[WebSearchResult]:
        """Process search results and convert them to content blocks.

        Args:
            query: The original search query that generated these results.
            results: List of search results to process.
            model_id: The model id of the model to process.

        Returns:
            List of processed content blocks ready for consumption.
        """
        pass

    async def validate(self) -> None:  # noqa: B027
        """Validate the processor configuration.

        This method should check if the processor is correctly configured
        and able to process search results. It may raise exceptions if
        validation fails.
        """
        pass
