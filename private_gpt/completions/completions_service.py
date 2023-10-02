from dataclasses import dataclass

from injector import inject, singleton
from llama_index import ServiceContext
from llama_index.llms import CompletionResponseGen, MessageRole
from llama_index.llms.base import LLM, ChatMessage, ChatResponseGen
from llama_index.vector_stores.types import VectorStore


@singleton
@inject
@dataclass(kw_only=True)
class CompletionsService:
    llm: LLM
    service_context: ServiceContext
    vector_store: VectorStore

    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        return self.llm.stream_complete(prompt)

    def stream_chat(
        self, message: str, chat_history: list[ChatMessage] | None
    ) -> ChatResponseGen:
        messages = [
            *chat_history[:20],
            ChatMessage(content=message, role=MessageRole.USER),
        ]
        return self.llm.stream_chat(messages)
