from dataclasses import dataclass
from typing import TYPE_CHECKING

from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.schema import NodeWithScore

from private_gpt.llm.llm_service import LLMService
from private_gpt.node_store.node_store_service import NodeStoreService
from private_gpt.vector_store.vector_store_service import VectorStoreService

if TYPE_CHECKING:
    from llama_index.schema import RelatedNodeInfo


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
        self,
        llm_service: LLMService,
        vector_store_service: VectorStoreService,
        node_store_service: NodeStoreService,
    ) -> None:
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.node_store_service = node_store_service
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_service.vector_store,
            docstore=node_store_service.doc_store,
            index_store=node_store_service.index_store,
        )
        self.query_service_context = ServiceContext.from_defaults(
            llm=llm_service.llm, embed_model="local"
        )

    def _get_sibling_nodes_text(
        self, node_with_score: NodeWithScore, related_number: int, forward: bool = True
    ):
        explored_nodes_texts = []
        current_node = node_with_score.node
        for _ in range(related_number):
            explored_node_info: RelatedNodeInfo | None = (
                current_node.next_node if forward else current_node.prev_node
            )
            if explored_node_info is None:
                break

            explored_node = self.storage_context.docstore.get_node(
                explored_node_info.node_id
            )

            explored_nodes_texts.append(explored_node.get_content())
            current_node = explored_node

        return explored_nodes_texts

    async def retrieve_relevant_nodes(
        self, query: str, limit: int = 10, context_size: int = 0
    ) -> list[RetrievedNode]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            storage_context=self.storage_context,
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
                    previous_texts=self._get_sibling_nodes_text(
                        node, context_size, False
                    ),
                    next_texts=self._get_sibling_nodes_text(node, context_size),
                )
            )

        return retrieved_nodes
