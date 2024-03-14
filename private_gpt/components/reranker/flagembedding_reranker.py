import logging
from typing import (  # noqa: UP035, we need to keep the consistence with llamaindex
    List,
    Tuple,
)

from FlagEmbedding import FlagReranker  # type: ignore
from llama_index.core.bridge.pydantic import Field
from llama_index.core.indices.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class FlagEmbeddingRerankerComponent(BaseNodePostprocessor):
    """Reranker component.

    - top_n: Top N nodes to return.
    - cut_off: Cut off score for nodes.

    If the number of nodes with score > cut_off is <= top_n, then return top_n nodes.
    Otherwise, return all nodes with score > cut_off.
    """

    top_n: int = Field(10, description="Top N nodes to return.")
    cut_off: float = Field(0.0, description="Cut off score for nodes.")
    reranker: FlagReranker = Field(..., description="Flag Reranker model.")

    @classmethod
    def class_name(cls) -> str:
        return "FlagEmbeddingReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],  # noqa: UP006
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:  # noqa: UP006
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        logger.info("Postprocessing nodes with FlagEmbeddingReranker.")
        logger.info(f"top_n: {self.top_n}, cut_off: {self.cut_off}")

        query_str = query_bundle.query_str
        sentence_pairs: List[Tuple[str, str]] = []  # noqa: UP006
        for node in nodes:
            content = node.get_content()
            sentence_pairs.append((query_str, content))

        scores = self.reranker.compute_score(sentence_pairs)
        for i, node in enumerate(nodes):
            node.score = scores[i]

        # cut off nodes with low scores
        res = [node for node in nodes if (node.score or 0.0) > self.cut_off]
        if len(res) > self.top_n:
            logger.info(
                "Number of nodes with score > cut_off is > top_n, returning all nodes with score > cut_off."
            )
            return res

        logger.info(
            "Number of nodes with score > cut_off is <= top_n, returning top_n nodes."
        )
        return sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]
