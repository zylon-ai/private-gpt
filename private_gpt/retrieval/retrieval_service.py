from dataclasses import dataclass
from typing import List

from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms.base import (
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
)
from llama_index.schema import NodeWithScore, BaseNode, RelatedNodeInfo

from private_gpt.llm.llm_service import LLMService
from private_gpt.llm.vector_store_service import VectorStoreService


@dataclass
class RetrievedNode:
    score: float
    name: str
    text: str
    previous_texts: List[str] | None
    next_texts: List[str] | None


def _sort_nodes(node: NodeWithScore) -> float:
    return node.score


@singleton
class RetrievalService:
    @inject
    def __init__(
        self, llm_service: LLMService, vector_store_service: VectorStoreService
    ) -> None:
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.query_service_context = ServiceContext.from_defaults(
            llm=llm_service.llm, embed_model="local"
        )

    async def retrieve_relevant_nodes(
        self, query: str, limit: int = 10, context_size: int = 0
    ) -> List[RetrievedNode]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.query_service_context,
            show_progress=True,
        )
        nodes = index.as_retriever(similarity_top_k=limit).retrieve(query)
        nodes.sort(key=_sort_nodes, reverse=True)
        retrieved_nodes = []
        index.docstore.get_nodes()
        for node in nodes:
            retrieved_nodes.append(
                RetrievedNode(
                    score=node.score,
                    name=node.metadata["file_name"],
                    text=node.get_content(),
                    previous_texts=None,
                    next_texts=None,
                )
            )

        return retrieved_nodes
