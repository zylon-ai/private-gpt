import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any, cast

from injector import inject, singleton
from llama_index.core.schema import BaseNode, NodeWithScore

from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.ingest.transformations.sentence_tree_node_parser import (
    TokenTextSplitterWithoutStripping,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import TokenizerFn, get_tokenizer
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.tree_expansion.split_subtrees import (
    SplitSubtreeAlg,
)
from private_gpt.components.readers.nodes import NodeType, TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings


class ContentRequestLimitError(ValueError):
    pass


def _split_subtree_to_fit(
    subtree: TreeNode,
    max_length: int | None,
    tokenizer_fn: TokenizerFn | None,
) -> list[BaseNode]:
    if max_length is None or tokenizer_fn is None:
        return [subtree]

    content = subtree.get_content(TreeMetadataMode.LLM)
    if len(tokenizer_fn(content)) <= max_length:
        return [subtree]

    splitter_class = cast(Any, TokenTextSplitterWithoutStripping)
    splitter = splitter_class(
        chunk_size=max_length,
        chunk_overlap=0,
        tokenizer=tokenizer_fn,
        keep_whitespaces=True,
    )
    chunks = splitter.split_text(content)
    if not chunks:
        raise ContentRequestLimitError("Unable to split oversized document subtree")

    split_nodes = [
        TextNode(
            text=chunk,
            extra_info=dict(subtree.metadata),
            abs_idx=subtree.abs_idx,
            idx=subtree.idx,
        )
        for chunk in chunks
        if chunk
    ]
    if any(len(tokenizer_fn(node.text)) > max_length for node in split_nodes):
        raise ContentRequestLimitError(
            "Document subtree could not be split within the requested token limit"
        )
    return cast(list[BaseNode], split_nodes)


@singleton
class ContentService:
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
        self.vector_store_component = vector_store_component
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.node_store_component = node_store_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component
        self.max_content_nodes = settings.data.max_content_nodes
        self.max_content_artifacts = settings.data.max_content_artifacts
        self.max_content_depth = settings.data.max_content_depth
        self.max_content_response_bytes = settings.data.max_content_response_bytes
        self._content_limiter = asyncio.Semaphore(settings.data.max_content_concurrency)

    @asynccontextmanager
    async def content_slot(self) -> AsyncGenerator[None]:
        async with self._content_limiter:
            yield

    async def _format_nodes(
        self,
        nodes: list[NodeWithScore],
        generate_citations: bool = False,
        token_limit: int | None = None,
        tokenizer_fn: TokenizerFn | None = None,
    ) -> list[Any]:
        """Format nodes into chunks."""
        from private_gpt.components.prompts.prompt_builder import PromptBuilderService

        prompt_builder_service = get_global_injector().get(PromptBuilderService)

        def format_result(
            n: list[NodeWithScore],
        ) -> str:
            prompt, _ = prompt_builder_service.create_context_prompt(
                nodes=n,
                generate_citations=generate_citations,
                token_limit=token_limit,
                tokenizer_fn=tokenizer_fn if tokenizer_fn else get_tokenizer(),
            )
            return prompt.format() or "No content is available."

        from private_gpt.events.models import SourceBlock, TextBlock

        def build_blocks() -> list[Any]:
            formatted_result = format_result(nodes)
            return [
                SourceBlock.from_nodes(nodes),
                TextBlock(text=formatted_result),
            ]

        return await asyncio.to_thread(build_blocks)

    def _get_root_node(
        self,
        artifact: str,
        context_filter: ContextFilter,
    ) -> TreeNode | None:
        """Retrieve root node by finding any node and traversing to root."""
        nodes = self.node_store_component.filtered_nodes(
            context_filter.collection,
            [artifact],
            context_filter.metadata_filter,
            limit=1,
        )
        if not nodes:
            return None

        root_id = getattr(nodes[0], "root_id", None)
        if not isinstance(root_id, str) or not root_id:
            return None

        root_nodes = self.node_store_component.filtered_nodes(
            context_filter.collection,
            [artifact],
            context_filter.metadata_filter,
            node_ids=[root_id],
            limit=1,
        )
        return cast(TreeNode, root_nodes[0]) if root_nodes else None

    def _filter_tree_nodes(
        self,
        root: TreeNode,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
    ) -> Generator[str, None, None]:
        """Flatten tree and apply type filters."""
        all_nodes: list[TreeNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            if len(all_nodes) > self.max_content_nodes:
                raise ContentRequestLimitError(
                    f"Artifact exceeds the {self.max_content_nodes} node limit"
                )
            stack.extend(reversed(node.children))
        node_map = {node.id_: node for node in all_nodes}

        nodes_to_include: set[str] | None = None

        if node_ids:
            node_ids_set = set(node_ids)
            nodes_to_include = set()

            for node_id in node_ids_set:
                if node_id in node_map:
                    node = node_map[node_id]
                    nodes_to_include.add(node_id)

                    if include_children:
                        descendants = list(reversed(node.children))
                        while descendants:
                            descendant = descendants.pop()
                            nodes_to_include.add(descendant.id_)
                            descendants.extend(reversed(descendant.children))

                    if include_ancestors:
                        current_node = node
                        while current_node.parent_id:
                            nodes_to_include.add(current_node.parent_id)
                            current_node = node_map[current_node.parent_id]

        for node in all_nodes:
            if nodes_to_include is not None and node.id_ not in nodes_to_include:
                continue
            if include is not None and not node.isinstance(tuple(include)):
                continue
            if exclude is not None and node.isinstance(tuple(exclude)):
                continue
            yield node.id_

    def _get_nodes_for_artifact(
        self,
        artifact: str,
        context_filter: ContextFilter,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
    ) -> list[TreeNode]:
        """Retrieve nodes for artifact, with optional type filtering."""
        has_filter = include or exclude or node_ids
        if not has_filter:
            # Optimization: if no type filters,
            # retrieve all nodes for artifact in one call
            return [
                cast(TreeNode, node)
                for node in self.node_store_component.filtered_nodes(
                    context_filter.collection,
                    [artifact],
                    context_filter.metadata_filter,
                    limit=self.max_content_nodes + 1,
                )
            ]

        # Otherwise, retrieve root and filter in memory
        # the partial tree (to avoid loading all nodes in memory if not needed)
        root = self._get_root_node(artifact=artifact, context_filter=context_filter)
        if not root:
            return []

        filtered_nodes_ids = list(
            self._filter_tree_nodes(
                root=root,
                include=include,
                exclude=exclude,
                node_ids=node_ids,
                include_children=include_children,
                include_ancestors=include_ancestors,
            )
        )
        if not filtered_nodes_ids:
            return []

        final_node_ids = set(filtered_nodes_ids)
        if root.id_ not in final_node_ids:
            final_node_ids.add(root.id_)

        nodes: list[BaseNode] = self.node_store_component.filtered_nodes(
            context_filter.collection,
            [artifact],
            context_filter.metadata_filter,
            node_ids=list(final_node_ids),
        )
        return [cast(TreeNode, node) for node in nodes]

    def _retrieve_document_node(
        self,
        context_filter: ContextFilter,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
    ) -> Generator[tuple[str, TreeNode], None, None]:
        collection = context_filter.collection

        # List unique root nodes
        artifacts: list[str] = context_filter.artifacts or []
        if not artifacts:
            artifacts = self.node_store_component.get_list_of_artifact_ids(collection)
        artifacts = list(set(artifacts))
        if len(artifacts) > self.max_content_artifacts:
            raise ContentRequestLimitError(
                f"Content request exceeds the {self.max_content_artifacts} artifact limit"
            )

        # If artifacts are provided, verify the related required indexes are ready
        # or throw an error
        if artifacts:
            for artifact in artifacts:
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

        # Get all nodes for each artifact
        for artifact in artifacts:
            nodes = self._get_nodes_for_artifact(
                context_filter=context_filter,
                artifact=artifact,
                include=include if include else None,
                exclude=exclude if exclude else None,
                node_ids=node_ids,
                include_children=include_children if node_ids else False,
                include_ancestors=include_ancestors if node_ids else False,
            )
            if not nodes:
                continue
            if len(nodes) > self.max_content_nodes:
                raise ContentRequestLimitError(
                    f"Artifact {artifact} exceeds the {self.max_content_nodes} node limit"
                )

            # Sort nodes by their position in the tree
            nodes = sorted(
                nodes,
                key=lambda n: n.abs_idx,
            )

            # Rebuilt the tree structure
            root_nodes = TreeNode.rebuild_tree(nodes)
            root_node = root_nodes[0] if root_nodes else None
            if not root_node:
                continue
            if max((node.depth for node in nodes), default=0) > self.max_content_depth:
                raise ContentRequestLimitError(
                    f"Artifact {artifact} exceeds the {self.max_content_depth} depth limit"
                )

            yield artifact, root_node

            # Cleanup
            del root_node
            del nodes

    async def retrieve_document_nodes_async(
        self,
        context_filter: ContextFilter,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
    ) -> AsyncGenerator[tuple[str, TreeNode]]:
        iterator = self._retrieve_document_node(
            context_filter=context_filter,
            include=include,
            exclude=exclude,
            node_ids=node_ids,
            include_children=include_children,
            include_ancestors=include_ancestors,
        )
        sentinel = object()

        async def iterate() -> AsyncGenerator[tuple[str, TreeNode]]:
            while True:
                item = await asyncio.to_thread(next, iterator, sentinel)
                if item is sentinel:
                    break
                yield cast(tuple[str, TreeNode], item)

        return iterate()

    def retrieve_document_content(
        self,
        context_filter: ContextFilter,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
    ) -> Generator[tuple[str, TreeNode], None, None]:
        """Retrieve document nodes based on the context filter."""
        for artifact, root_node in self._retrieve_document_node(
            context_filter=context_filter,
            include=include,
            exclude=exclude,
            node_ids=node_ids,
            include_children=include_children,
            include_ancestors=include_ancestors,
        ):
            yield artifact, root_node

            # Cleanup
            del root_node

    async def retrieve_chunked_document_content(
        self,
        context_filter: ContextFilter,
        include: list[type[NodeType]] | None = None,
        exclude: list[type[NodeType]] | None = None,
        node_ids: list[str] | None = None,
        include_children: bool = True,
        include_ancestors: bool = False,
        generate_citations: bool = False,
        max_length: int | None = None,
        tokenizer_fn: TokenizerFn | None = None,
    ) -> AsyncGenerator[tuple[str, list[Any]]]:
        """Retrieve chunked document content based on the context filter."""
        current_index = 0

        async def _format_nodes(tree_nodes: list[BaseNode]) -> list[Any]:
            nonlocal current_index

            from private_gpt.components.engines.citations.utils import (
                exclude_metadata,
                init_nodes_with_shorter_ids,
            )

            # Shorten node IDs and exclude metadata
            nodes = [NodeWithScore(node=n, score=0.0) for n in tree_nodes]
            nodes = init_nodes_with_shorter_ids(nodes, initial_index=current_index)
            nodes = exclude_metadata(nodes)
            current_index += len(nodes)

            return await self._format_nodes(
                nodes=nodes,
                generate_citations=generate_citations,
                token_limit=max_length,
                tokenizer_fn=tokenizer_fn,
            )

        async for artifact, root_node in await self.retrieve_document_nodes_async(
            context_filter=context_filter,
            include=include,
            exclude=exclude,
            node_ids=node_ids,
            include_children=include_children,
            include_ancestors=include_ancestors,
        ):
            # Split the tree into subtrees
            alg: SplitSubtreeAlg = SplitSubtreeAlg()
            subtrees = await asyncio.to_thread(
                alg.split_subtree,
                root_node,
            )

            def build_chunks(
                subtrees_for_artifact: list[TreeNode] = subtrees,
            ) -> list[list[BaseNode]]:
                chunks: list[list[BaseNode]] = []
                current_chunk: list[BaseNode] = []
                current_chunk_tokens = 0
                bounded_nodes = [
                    bounded_node
                    for subtree in subtrees_for_artifact
                    for bounded_node in _split_subtree_to_fit(
                        subtree,
                        max_length,
                        tokenizer_fn,
                    )
                ]
                for bounded_node in bounded_nodes:
                    if not isinstance(bounded_node, TreeNode):
                        raise TypeError("Chunked document nodes must be tree nodes")
                    bounded_content = (
                        bounded_node.text
                        if isinstance(bounded_node, TextNode)
                        and not bounded_node.children
                        else bounded_node.get_content(TreeMetadataMode.LLM)
                    )
                    bounded_tokens = (
                        len(tokenizer_fn(bounded_content)) if tokenizer_fn else None
                    )

                    if (
                        current_chunk
                        and max_length is not None
                        and bounded_tokens is not None
                        and current_chunk_tokens + bounded_tokens > max_length
                    ):
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_chunk_tokens = 0

                    current_chunk.append(bounded_node)
                    if bounded_tokens is not None:
                        current_chunk_tokens += bounded_tokens

                if current_chunk:
                    chunks.append(current_chunk)
                return chunks

            chunks = await asyncio.to_thread(build_chunks)
            for chunk in chunks:
                yield artifact, await _format_nodes(chunk)

            # Cleanup
            del root_node
            del subtrees
            del chunks
