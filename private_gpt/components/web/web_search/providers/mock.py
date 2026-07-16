import logging
from typing import Any

from injector import singleton

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider

logger = logging.getLogger(__name__)


@singleton
class MockSearchProvider(BaseWebSearchProvider):
    def __init__(self) -> None:
        self._requests_count = 0

    async def make_query(
        self, query: str, num_links: int = 10, **kwargs: Any
    ) -> list[WebSearchResult]:
        self._requests_count += 1
        mock_results = [
            WebSearchResult(
                idx=1,
                title="Zylon | The On-premise Private AI platform for Regulated Industries",
                url="https://www.zylon.ai/",
                description="Zylon is <strong>an on-premise alternative to AI tools like ChatGPT</strong> that runs on your Virtual Private Cloud (AWS, Azure, GCP...) or local data center (even air-gapped) behind your firewall.",
                age="100 days ago",
            ),
            WebSearchResult(
                idx=2,
                title="Zylon - Wikipedia",
                url="https://en.wikipedia.org/wiki/Zylon",
                description="Zylon (IUPAC name: poly(p-phenylene-2,6-benzobisoxazole)) is <strong>a trademarked name for a range of thermoset liquid-crystalline polyoxazole</strong>. This synthetic polymer material was invented and developed by SRI International in the 1980s and manufactured ...",
                age="2 weeks ago",
            ),
            WebSearchResult(
                idx=3,
                title="ZYLON® | Products | Toyobo MC Corporation",
                url="https://en.toyobo-mc.jp/products/zylon/",
                description="Zylon® consists of rigid rod chain molecules of poly (p-phenylene-2, 6-benzbisoxazole (PBO).",
                age="",
            ),
            WebSearchResult(
                idx=4,
                title="GitHub - zylon-ai/private-gpt: Interact with your documents using the power of GPT, 100% privately, no data leaks",
                url="https://github.com/zylon-ai/private-gpt",
                description="Crafted by the team behind PrivateGPT, Zylon is <strong>a best-in-class AI collaborative workspace</strong> that can be easily deployed on-premise (data center, bare metal...) or in your private cloud (AWS, GCP, Azure...).",
                age="",
            ),
            WebSearchResult(
                idx=5,
                title="75 Years of Innovation: Synthetic polymer, Zylon - SRI",
                url="https://www.sri.com/press/story/75-years-of-innovation-synthetic-polymer-zylon/",
                description="A reaction occurs. The teacher inserts a glass rod into the mixture and pulls out a long string of plastic, specifically Nylon polymer. Nylon was first developed in the 1930s. Fifty years later, SRI International developed a new polymer that was given the name Zylon.",
                age="March 19, 2025",
            ),
        ]

        results = mock_results[:num_links]
        logger.debug(
            f"Mock search completed for query: '{query}' - {len(results)} results"
        )
        return results

    def get_rate_limit_info(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "requests_made": self._requests_count,
            "is_rate_limited": False,
        }

    async def close(self) -> None:
        logger.debug("MockSearchProvider closed")
