from abc import ABC, abstractmethod
from typing import Any

from private_gpt.components.web.web_search.models import WebSearchResult


class BaseWebSearchProvider(ABC):
    """Base class for web search providers.

    This abstract class defines the interface that all web search providers
    must implement. Providers are responsible for connecting to external
    search APIs and returning standardized search results.
    """

    @abstractmethod
    async def make_query(
        self, query: str, num_links: int, **kwargs: Any
    ) -> list[WebSearchResult]:
        """Execute a search query and return results.

        Args:
            query: The search query string.
            num_links: The number of search results to return.
            **kwargs: Provider-specific parameters (e.g., count, offset,
                     language, filters, etc.).

        Returns:
            List of standardized search results from the provider.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close and cleanup provider resources.

        This method should be called when the provider is no longer needed
        to properly release resources like HTTP sessions, connections, or
        file handles.
        """
        pass

    async def validate(self) -> None:  # noqa: B027
        """Validate the provider configuration.

        This method should check if the provider is correctly configured
        and able to connect to the external search API. It may raise
        exceptions if validation fails.
        """
        pass
