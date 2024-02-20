from typing import List, Tuple
from injector import singleton, inject
from llama_index.schema import NodeWithScore, QueryBundle
from private_gpt.paths import models_path
from llama_index.bridge.pydantic import Field
from FlagEmbedding import FlagReranker
from llama_index.postprocessor.types import BaseNodePostprocessor
from private_gpt.settings.settings import Settings


@singleton
class RerankerComponent(BaseNodePostprocessor):
    """
    Reranker component:
    - top_n: Top N nodes to return.
    - cut_off: Cut off score for nodes.

    If the number of nodes with score > cut_off is <= top_n, then return top_n nodes.
    Otherwise, return all nodes with score > cut_off.
    """

    reranker: FlagReranker = Field(description="Reranker class.")
    top_n: int = Field(description="Top N nodes to return.")
    cut_off: float = Field(description="Cut off score for nodes.")

    @inject
    def __init__(self, settings: Settings) -> None:
        if settings.reranker.enabled is False:
            raise ValueError("Reranker component is not enabled.")

        path = models_path / "reranker"
        top_n = settings.reranker.top_n
        cut_off = settings.reranker.cut_off
        reranker = FlagReranker(
            model_name_or_path=path,
        )

        super().__init__(
            top_n=top_n,
            reranker=reranker,
            cut_off=cut_off,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Reranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            return ValueError("Query bundle must be provided.")

        query_str = query_bundle.query_str
        sentence_pairs: List[Tuple[str, str]] = []
        for node in nodes:
            content = node.get_content()
            sentence_pairs.append([query_str, content])

        scores = self.reranker.compute_score(sentence_pairs)
        for i, node in enumerate(nodes):
            node.score = scores[i]

        # cut off nodes with low scores
        res = [node for node in nodes if node.score > self.cut_off]
        if len(res) > self.top_n:
            return res

        return sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]
