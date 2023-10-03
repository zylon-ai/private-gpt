from dataclasses import dataclass

from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex

from private_gpt.llm.llm_service import LLMService
from private_gpt.llm.vector_store_service import VectorStoreService


@dataclass
class RetrievedNode:
    score: float
    name: str
    text: str
    previous_texts: list[str] | None
    next_texts: list[str] | None


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
    ) -> list[RetrievedNode]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.query_service_context,
            show_progress=True,
        )
        nodes = await index.as_retriever(similarity_top_k=limit).aretrieve(query)
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
        retrieved_nodes = []
        for node in nodes:
            retrieved_nodes.append(
                RetrievedNode(
                    score=node.score or 0.0,
                    name=node.metadata["file_name"],
                    text=node.get_content(),
                    previous_texts=None,
                    next_texts=None,
                )
            )

        return retrieved_nodes
