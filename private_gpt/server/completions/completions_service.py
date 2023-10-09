from typing import TYPE_CHECKING, Any

from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.llm_predictor.utils import stream_completion_response_to_tokens
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.types import TokenGen

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter

if TYPE_CHECKING:
    from llama_index.response import Response
    from llama_index.response.schema import StreamingResponse


@singleton
class CompletionsService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.service_context = ServiceContext.from_defaults(
            llm=llm_component.llm, embed_model=embedding_component.embedding_model
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )

    def _complete_with_contex(
        self, prompt: str, context_filter: ContextFilter, streaming: bool = False
    ) -> Any:
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=self.index, context_filter=context_filter
        )
        query_engine = RetrieverQueryEngine.from_args(
            retriever=vector_index_retriever,
            service_context=self.service_context,
            streaming=streaming,
        )
        return query_engine.query(prompt)

    def stream_complete(
        self,
        prompt: str,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> TokenGen:
        if use_context:
            response: StreamingResponse = self._complete_with_contex(
                prompt, context_filter, True
            )
            response_gen = response.response_gen
        else:
            stream = self.llm_service.llm.stream_complete(prompt)
            response_gen = stream_completion_response_to_tokens(stream)
        return response_gen

    def complete(
        self,
        prompt: str,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> str:
        if use_context:
            complete_response: Response = self._complete_with_contex(
                prompt, context_filter, False
            )
            complete_text = complete_response.response
            response = complete_text if complete_text is not None else ""
        else:
            completion_response = self.llm_service.llm.complete(prompt)
            completion_text = completion_response.text
            response = completion_text if completion_text is not None else ""
        return response
