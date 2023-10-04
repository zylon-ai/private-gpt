from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.chat_engine import ContextChatEngine
from llama_index.chat_engine.types import ChatMode
from llama_index.llms.base import (
    ChatMessage,
    ChatResponseGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.types import TokenGen

from private_gpt.llm.llm_service import LLMService
from private_gpt.vector_store.vector_store_service import VectorStoreService


@singleton
class QueryService:
    @inject
    def __init__(
        self, llm_service: LLMService, vector_store_service: VectorStoreService
    ) -> None:
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.query_service_context = ServiceContext.from_defaults(
            llm=llm_service.llm, embed_model="local"
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.query_service_context,
            show_progress=True,
        )

    def stream_complete(self, prompt: str) -> TokenGen:
        query_engine = self.index.as_query_engine(streaming=True)
        return query_engine.query(prompt).response_gen

    def stream_chat(
        self, query: str, history: list[ChatMessage] | None = None
    ) -> TokenGen:
        context_chat_engine = self.index.as_chat_engine(
            chat_mode=ChatMode.BEST, streaming=True
        )
        result = context_chat_engine.stream_chat(query, history)
        return result.response_gen
