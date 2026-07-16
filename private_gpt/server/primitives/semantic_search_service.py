from typing import cast

from injector import inject, singleton
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.chunk.models import Chunk
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.tree_expansion.document_expander import (
    DocumentTreeExpander,
)
from private_gpt.components.readers.nodes import TreeNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.settings.settings import Settings


@singleton
class SemanticSearchService:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        ingest_component: IngestComponent,
        parse_component: ParseComponent,
    ) -> None:
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component
        self.embedding_component = embedding_component
        self.node_store_component = node_store_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component

    def _expand_nodes(
        self,
        nodeWithScore: NodeWithScore,
        collection: str,
        token_limit: int,
    ) -> tuple[list[str], list[str]]:
        hit_node = nodeWithScore.node
        if not isinstance(hit_node, TreeNode):
            return [], []

        root_nodes = self.node_store_component.get_sorted_nodes(
            collection=collection,
            node_ids=[hit_node.root_id or hit_node.id_],
        )
        if not root_nodes or not isinstance(root_nodes[0], TreeNode):
            return [], []

        partial_hit_node = root_nodes[0].find_self_or_child_by_id(hit_node.id_)
        if not partial_hit_node:
            return [], []

        alg: DocumentTreeExpander = DocumentTreeExpander(
            partial_hit_node,
            token_limit,
        )
        unsorted_nodes_ids, _ = alg.fill_window()
        sorted_nodes = self.node_store_component.get_sorted_nodes(
            collection, node_ids=list(unsorted_nodes_ids)
        )

        hit_node_idx = next(
            (i for i, node in enumerate(sorted_nodes) if node.id_ == hit_node.id_), -1
        )
        if hit_node_idx == -1:
            return [], []
        prev_nodes = sorted_nodes[:hit_node_idx]
        next_nodes = sorted_nodes[hit_node_idx + 1 :]

        prev_texts = [
            cast(TreeNode, node).get_content(TreeMetadataMode.USER)
            for node in prev_nodes
        ]
        next_text = [
            cast(TreeNode, node).get_content(TreeMetadataMode.USER)
            for node in next_nodes
        ]
        return prev_texts, next_text

    def retrieve_semantic_relevant(
        self,
        text: str,
        context_filter: ContextFilter,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        expand: bool = True,
        token_limit: int | None = None,
        validate: bool = True,
    ) -> list[Chunk]:
        llm = self.llm_component.get_llm(model_id)
        collection = context_filter.collection
        # If artifacts are provided, verify the related required indexes are ready
        # or throw an error
        if context_filter.artifacts and validate:
            for artifact in context_filter.artifacts:
                vector_artifact_index = VectorArtifactIndex(
                    collection=collection,
                    artifact=artifact,
                    vector_store_component=self.vector_store_component,
                    node_store_component=self.node_store_component,
                    embedding_component=self.embedding_component,
                    ingest_component=self.ingest_component,
                    parse_component=self.parse_component,
                )
                vector_artifact_index.populated_or_error()

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store_component.vector_store(collection),
            index_store=self.node_store_component.index_store(collection),
        )

        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store(collection),
            storage_context=storage_context,
            llm=llm,
            embed_model=self.embedding_component.get_embed(embed_model_id),
            show_progress=False,
        )
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=index,
            artifacts=context_filter.artifacts,
            collection=collection,
            filter_dicts=context_filter.metadata_filter,
            similarity_top_k=limit,
            score_threshold=score_threshold,
        )
        nodes = vector_index_retriever.retrieve(text)
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        retrieved_nodes = []
        final_token_limit = token_limit or llm.metadata.context_window // 8
        for node in nodes:
            chunk = Chunk.from_node(node)
            if expand:
                chunk.previous_texts, chunk.next_texts = self._expand_nodes(
                    node, collection, token_limit=final_token_limit
                )
            retrieved_nodes.append(chunk)

        return retrieved_nodes
