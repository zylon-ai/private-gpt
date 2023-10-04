from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms.base import (
    ChatMessage,
    ChatResponseGen,
    CompletionResponseGen,
    MessageRole,
)

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

    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        return self.llm_service.stream_complete(prompt)

    def stream_chat(
        self, query: str, history: list[ChatMessage] | None = None
    ) -> ChatResponseGen:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.query_service_context,
            show_progress=True,
        )
        nodes = index.as_retriever().retrieve(query)
        nodes_content = [node.get_content() for node in nodes]
        context = "\n\n".join(nodes_content)

        system_message_content = "\n".join(
            [
                "Answer questions using the information below when relevant:",
                "--------",
                context,
                "--------",
            ]
        )

        system_message = ChatMessage(
            content=system_message_content, role=MessageRole.SYSTEM
        )

        if history is None:
            history = []
        history = [system_message, *history]

        return self.llm_service.stream_chat(query, history)
