from typing import TYPE_CHECKING, Literal

from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.schema import NodeWithScore
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.ingest.model import IngestedDoc

if TYPE_CHECKING:
    from llama_index.schema import RelatedNodeInfo


class Chunk(BaseModel):
    object: Literal["context.chunk"]
    score: float = Field(examples=[0.023])
    document: IngestedDoc
    text: str = Field(examples=["Outbound sales increased 20%, driven by new leads."])
    previous_texts: list[str] | None = Field(
        default=None,
        examples=[["SALES REPORT 2023", "Inbound didn't show major changes."]],
    )
    next_texts: list[str] | None = Field(
        default=None,
        examples=[
            [
                "New leads came from Google Ads campaign.",
                "The campaign was run by the Marketing Department",
            ]
        ],
    )

    @classmethod
    def from_node(cls: type["Chunk"], node: NodeWithScore) -> "Chunk":
        doc_id = node.node.ref_doc_id if node.node.ref_doc_id is not None else "-"
        return cls(
            object="context.chunk",
            score=node.score or 0.0,
            document=IngestedDoc(
                object="ingest.document",
                doc_id=doc_id,
                doc_metadata=node.metadata,
            ),
            text=node.get_content(),
        )


@singleton
class ChunksService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.query_service_context = ServiceContext.from_defaults(
            llm=llm_component.llm, embed_model=embedding_component.embedding_model
        )

    def _get_sibling_nodes_text(
        self, node_with_score: NodeWithScore, related_number: int, forward: bool = True
    ) -> list[str]:
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

    def retrieve_relevant(
        self,
        text: str,
        context_filter: ContextFilter | None = None,
        limit: int = 10,
        prev_next_chunks: int = 0,
    ) -> list[Chunk]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store,
            storage_context=self.storage_context,
            service_context=self.query_service_context,
            show_progress=True,
        )
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=index, context_filter=context_filter, similarity_top_k=limit
        )
        nodes = vector_index_retriever.retrieve(text)
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        retrieved_nodes = []
        for node in nodes:
            chunk = Chunk.from_node(node)
            chunk.previous_texts = self._get_sibling_nodes_text(
                node, prev_next_chunks, False
            )
            chunk.next_texts = self._get_sibling_nodes_text(node, prev_next_chunks)
            retrieved_nodes.append(chunk)

        return retrieved_nodes
