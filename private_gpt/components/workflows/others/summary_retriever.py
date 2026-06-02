from abc import ABC, abstractmethod
from asyncio import Protocol
from collections.abc import AsyncGenerator, Generator, Sequence

from llama_index.core import QueryBundle
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.readers.nodes.frozen_node import FrozenNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode
from private_gpt.server.content.content_service import ContentService


class Retriever(ABC, Protocol):
    @abstractmethod
    def retrieve(
        self, query_bundle: QueryBundle
    ) -> Generator[NodeWithScore, None, None]:
        pass

    @abstractmethod
    async def aretriever(
        self, query_bundle: QueryBundle
    ) -> AsyncGenerator[NodeWithScore]:
        pass


class InMemoryRetriever(Retriever):
    _nodes: list[NodeWithScore]

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self._nodes = nodes

    @classmethod
    def from_nodes(cls, nodes: list[BaseNode]) -> "InMemoryRetriever":
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        return cls(node_with_scores)

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> "InMemoryRetriever":
        nodes = [TextNode(text=text) for text in texts]
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        return cls(node_with_scores)

    def retrieve(
        self, query_bundle: QueryBundle
    ) -> Generator[NodeWithScore, None, None]:
        yield from self._nodes

    async def aretriever(
        self, query_bundle: QueryBundle
    ) -> AsyncGenerator[NodeWithScore]:
        async def gen() -> AsyncGenerator[NodeWithScore]:
            for node in self._nodes:
                yield node

        return gen()


class ContextRetriever(Retriever):
    _content_service: ContentService
    _context_filter: ContextFilter

    def __init__(
        self,
        content_service: ContentService,
        context_filter: ContextFilter,
    ) -> None:
        self._content_service = content_service
        self._context_filter = context_filter

    def retrieve(
        self, query_bundle: QueryBundle
    ) -> Generator[NodeWithScore, None, None]:
        context_documents_gen = self._content_service.retrieve_document_content(
            context_filter=self._context_filter,
        )
        for _, root_node in context_documents_gen:
            frozen_node = FrozenNode.from_node(root_node, modes=[TreeMetadataMode.LLM])
            node_with_score = NodeWithScore(node=frozen_node)
            yield node_with_score
            del node_with_score

    async def aretriever(
        self, query_bundle: QueryBundle
    ) -> AsyncGenerator[NodeWithScore, None]:
        async def gen() -> AsyncGenerator[NodeWithScore, None]:
            for node in self.retrieve(query_bundle):
                yield node

        return gen()


class CompositeRetriever(Retriever):
    _retrievers: list[Retriever]

    def __init__(self, retrievers: list[Retriever]) -> None:
        self._retrievers = retrievers

    def retrieve(
        self, query_bundle: QueryBundle
    ) -> Generator[NodeWithScore, None, None]:
        for retriever in self._retrievers:
            yield from retriever.retrieve(query_bundle)

    async def aretriever(
        self, query_bundle: QueryBundle
    ) -> AsyncGenerator[NodeWithScore]:
        async def gen() -> AsyncGenerator[NodeWithScore]:
            for retriever in self._retrievers:
                gen_coro = await retriever.aretriever(query_bundle)
                async for node in gen_coro:
                    yield node

        return gen()
