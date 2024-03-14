from typing import (  # noqa: UP035, we need to keep the consistence with llamaindex
    List,
    Tuple,
)

from FlagEmbedding import FlagReranker  # type: ignore
from llama_index.core.bridge.pydantic import Field
from llama_index.core.indices.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings


class FlagEmbeddingRerankerComponent(BaseNodePostprocessor):
    """Reranker component.

    - top_n: Top N nodes to return.
    - cut_off: Cut off score for nodes.

    If the number of nodes with score > cut_off is <= top_n, then return top_n nodes.
    Otherwise, return all nodes with score > cut_off.
    """

    reranker: FlagReranker = Field(description="Reranker class.")
    top_n: int = Field(description="Top N nodes to return.")
    cut_off: float = Field(description="Cut off score for nodes.")

    def __init__(self, settings: Settings) -> None:
        path = models_path / "flagembedding_reranker"
        top_n = settings.flagembedding_reranker.top_n
        cut_off = settings.flagembedding_reranker.cut_off
        reranker = FlagReranker(
            model_name_or_path=path,
        )

        super().__init__(
            top_n=top_n,
            cut_off=cut_off,
            reranker=reranker,
        )

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
            return res

        return sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]
