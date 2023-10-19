from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from injector import inject, singleton
from llama_index import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.chat_engine import ContextChatEngine
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llm_predictor.utils import stream_chat_response_to_tokens
from llama_index.llms import ChatMessage
from llama_index.types import TokenGen

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter

if TYPE_CHECKING:
    from llama_index.chat_engine.types import (
        AgentChatResponse,
        StreamingAgentChatResponse,
    )


@singleton
class ChatService:
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

    def _chat_with_contex(
        self,
        message: str,
        context_filter: ContextFilter | None = None,
        chat_history: Sequence[ChatMessage] | None = None,
        streaming: bool = False,
    ) -> Any:
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=self.index, context_filter=context_filter
        )
        chat_engine = ContextChatEngine.from_defaults(
            retriever=vector_index_retriever,
            service_context=self.service_context,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ],
        )
        if streaming:
            result = chat_engine.stream_chat(message, chat_history)
        else:
            result = chat_engine.chat(message, chat_history)
        return result

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> TokenGen:
        if use_context:
            last_message = messages[-1].content
            response: StreamingAgentChatResponse = self._chat_with_contex(
                message=last_message if last_message is not None else "",
                chat_history=messages[:-1],
                context_filter=context_filter,
                streaming=True,
            )
            response_gen = response.response_gen
        else:
            stream = self.llm_service.llm.stream_chat(messages)
            response_gen = stream_chat_response_to_tokens(stream)
        return response_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> str:
        if use_context:
            last_message = messages[-1].content
            wrapped_response: AgentChatResponse = self._chat_with_contex(
                message=last_message if last_message is not None else "",
                chat_history=messages[:-1],
                context_filter=context_filter,
                streaming=False,
            )
            response = wrapped_response.response
        else:
            chat_response = self.llm_service.llm.chat(messages)
            response_content = chat_response.message.content
            response = response_content if response_content is not None else ""
        return response
