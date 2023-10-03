from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms.base import (
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
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

    async def stream_complete(self, prompt: str) -> CompletionResponseAsyncGen:
        return await self.llm_service.stream_complete(prompt)

    async def stream_chat(self, query: str) -> ChatResponseAsyncGen:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.query_service_context,
            show_progress=True,
        )
        nodes = index.as_retriever().retrieve(query)
        nodes_content = [node.get_content() for node in nodes]
        context = "\n\n".join(nodes_content)
        print(context)
        message = (
            "Here is some context:\n"
            + context
            + "\n\n Using only the previous context, answer the following question. "
            "Just answer directly without explaining how you got to it. Here is the question: "
            + query
        )
        return await self.llm_service.stream_chat(message)
