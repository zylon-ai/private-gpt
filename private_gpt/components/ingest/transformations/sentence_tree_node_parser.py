"""Simple node parser."""
import copy
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser.interface import NodeParser, TextSplitter
from llama_index.core.node_parser.node_utils import (
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.readers.nodes.chunk_node import ChunkNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeNode
from private_gpt.di import get_global_injector

DEFAULT_WINDOW_SIZE = 3
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


def contains_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any(1536 <= ord(c) <= 1791 for c in text[:50])


def split_by_regex_sentence_internal(text: str) -> list[str]:
    pattern = r"(?<=[\.!\?؟])\s+"
    sentences = re.split(pattern, text.strip())
    return sentences


def split_by_nltk_sentence_internal(text: str) -> list[str]:
    from llama_index.core.node_parser.text.utils import (
        split_by_sentence_tokenizer_internal,
    )
    from nltk.tokenize import PunktSentenceTokenizer

    tokenizer = PunktSentenceTokenizer()
    return split_by_sentence_tokenizer_internal(text, tokenizer)


def split_by_sentence_tokenizer_internal(text: str) -> list[str]:
    sentences = split_by_nltk_sentence_internal(text)
    if not contains_arabic(text):
        return sentences
    arabic_sentences = split_by_regex_sentence_internal(text)
    return sentences if len(sentences) > len(arabic_sentences) else arabic_sentences


def split_by_sentence_tokenizer() -> Callable[[str], list[str]]:
    """Get a function that splits text into sentences."""
    return lambda text: split_by_sentence_tokenizer_internal(text)


class TokenTextSplitterWithoutStripping(TokenTextSplitter):
    def _merge(self, splits: list[str], chunk_size: int) -> list[str]:
        """Merge splits into chunks.

        The high-level idea is to keep adding splits to a chunk until we
        exceed the chunk size, then we start a new chunk with overlap.

        When we start a new chunk, we pop off the first element of the previous
        chunk until the total length is less than the chunk size.
        """
        chunks: list[str] = []

        cur_chunk: list[str] = []
        cur_len = 0
        for split in splits:
            split_len = len(self._tokenizer(split))
            if split_len > chunk_size:
                logging.warning(
                    f"Got a split of size {split_len}, ",
                    f"larger than chunk size {chunk_size}.",
                )

            # if we exceed the chunk size after adding the new split, then
            # we need to end the current chunk and start a new one
            if cur_len + split_len > chunk_size:
                # end the previous chunk
                chunk = "".join(cur_chunk)
                if chunk:
                    chunks.append(chunk)

                # start a new chunk with overlap
                # keep popping off the first element of the previous chunk until:
                #   1. the current chunk length is less than chunk overlap
                #   2. the total length is less than chunk size
                while cur_len > self.chunk_overlap or cur_len + split_len > chunk_size:
                    # pop off the first element
                    first_chunk = cur_chunk.pop(0)
                    cur_len -= len(self._tokenizer(first_chunk))

            cur_chunk.append(split)
            cur_len += split_len

        # handle the last chunk
        chunk = "".join(cur_chunk)
        if chunk:
            chunks.append(chunk)

        return chunks


class SentenceTreeNodeParser(NodeParser):
    """Sentence node parser.

    Splits a document into Nodes, with each node being a sentence.

    Args:
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    sentence_splitter: Callable[[str], list[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )
    fallback_text_splitter: TextSplitter | None = Field(
        default=None,
        description="Fallback text splitter to use when sentence splitter fails.",
    )
    copy: ClassVar = copy

    @classmethod
    def class_name(cls) -> str:
        return "SentenceWindowNodeParser"

    @classmethod
    def from_defaults(
        cls,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        fallback_text_splitter: TextSplitter | None = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
        id_func: Callable[[int, Document], str] | None = None,
    ) -> "SentenceTreeNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        embedding_component = get_global_injector().get(EmbeddingComponent)
        embed_context_window = embedding_component.get_config().context_window
        fallback_text_splitter = fallback_text_splitter or TokenTextSplitterWithoutStripping(
            # Config:
            # 1. Chunk size should be half of the context window size to
            #    allow to add metadata to the original text.
            # 2. Chunk overlap should be 0 to avoid overlapping chunks.
            #    If we overlap content, it will store the same content in couple chunks.
            chunk_size=int(embed_context_window * 0.9),
            chunk_overlap=0,
        )

        id_func = id_func or default_id_func

        return cls(
            sentence_splitter=sentence_splitter,
            fallback_text_splitter=fallback_text_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Parse nodes and convert nodes into sentences.

        Convert N TextNodes into P TextNodes, where P is the number
        of sentences in the original N TextNodes.
        This will create all relationships between the new nodes and the original nodes
            if include_prev_next_rel is True, linking all nodes in order,
            although they are from different Documents.
        Also, if include_metadata is True, the metadata
            from the original TextNode will be copied to all children.
        """

        def generate_chunks_in_tree(node: TreeNode) -> None:
            """Recursively traverse the tree and generate sentence chunks."""
            for child in node.children:
                generate_chunks_in_tree(child)

            # Process current node only if it's a leaf TextTreeNode
            if not isinstance(node, TextNode) or node.children:
                return

            # Split text into sentences and build new chunks
            text = node.text
            if not text.strip():
                return

            text_splits = self.sentence_splitter(text)
            if self.fallback_text_splitter:
                # If the maximum sentence length is applied,
                # split the sentences into max length chunks
                nested_texts = [
                    self.fallback_text_splitter.split_text(reduced_text_split)
                    for reduced_text_split in text_splits
                ]
                # Reduce the dimensions of the text_splits list
                text_splits = [item for sublist in nested_texts for item in sublist]

            if len(text_splits) > 1:
                node.text = ""

                # Create chunks for sentences/text_splits
                for i, text_chunk in enumerate(text_splits):
                    cloned_node = ChunkNode(**self.copy.deepcopy(node.dict()))
                    cloned_node.id_ = self.id_func(i, node)
                    cloned_node.text = text_chunk
                    cloned_node.parent = node
                    cloned_node.children = []
                    if self.include_metadata:
                        cloned_node.metadata = {
                            **(node.parent.metadata if node.parent else {}),
                            **cloned_node.metadata,
                        }
                    else:
                        cloned_node.metadata = {}
                    node.add_child(cloned_node)

        all_nodes: list[BaseNode] = []

        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for root_node in nodes_with_progress:
            if not isinstance(root_node, TreeNode):
                all_nodes.append(root_node)
                continue

            generate_chunks_in_tree(root_node)
            all_nodes.append(root_node)

        return all_nodes
