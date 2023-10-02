from dataclasses import dataclass

from injector import inject, singleton
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import CompletionResponseGen, MessageRole
from llama_index.llms.base import LLM, ChatMessage, ChatResponseGen
from llama_index.vector_stores.types import VectorStore


@singleton
@inject
@dataclass
class QueryService:
    llm: LLM
    service_context: ServiceContext
    vector_store: VectorStore

    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        return self.llm.stream_complete(prompt)

    def stream_chat(self, query: str) -> ChatResponseGen:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store, service_context=self.service_context, show_progress=True
        )
        nodes = index.as_retriever().retrieve(query)
        nodes_content = [node.get_content() for node in nodes]
        context = "\n\n".join(nodes_content)
        print(context)
        message = (
            "Here is some context:\n"
            + context
            + "\n\n Using only the previous context, answer this question: "
            + query
        )
        messages = [ChatMessage(content=message, role=MessageRole.USER)]
        return self.llm.stream_chat(messages)
