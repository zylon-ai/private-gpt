import logging
import time
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.llm.llm_helper import (
    TokenCountBatchFn,
    get_token_count_batch,
)
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

logger = logging.getLogger(__name__)

# Chunk size for batched tokenization. Large enough to keep the native
# batch/parallel encode path efficient, small enough to log progress on
# very large documents instead of blocking silently on one giant call.
_TOKEN_COUNT_BATCH_SIZE = 5000


class IncludeTokenCountIntoNodesTransform(TransformComponent):
    """Include token length in the nodes."""

    count_tokens_batch: TokenCountBatchFn | None

    @classmethod
    def from_defaults(
        cls, count_tokens_batch: TokenCountBatchFn | None = None
    ) -> "IncludeTokenCountIntoNodesTransform":
        return cls(count_tokens_batch=count_tokens_batch or get_token_count_batch())

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        if not self.count_tokens_batch:
            return nodes

        contents: list[str] = []
        contentful_nodes: list[BaseNode] = []
        for node in nodes:
            content = (
                node.get_content(TreeMetadataMode.RAG)
                if isinstance(node, TreeNode)
                else node.get_content(MetadataMode.NONE)
            )
            if not content:
                # Skip to avoid to save a token count of emtpy
                continue
            contents.append(content)
            contentful_nodes.append(node)

        if not contents:
            return nodes

        total = len(contents)
        logger.info("Counting tokens for %d nodes", total)
        t_start = time.perf_counter()

        for start in range(0, total, _TOKEN_COUNT_BATCH_SIZE):
            end = min(start + _TOKEN_COUNT_BATCH_SIZE, total)
            batch_counts = self.count_tokens_batch(contents[start:end])
            for node, token_count in zip(
                contentful_nodes[start:end], batch_counts, strict=True
            ):
                node.metadata[MetadataNode.TOKEN_COUNT.value] = token_count
                node.excluded_llm_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)
                node.excluded_embed_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)
            logger.info(
                "Token count progress: %d/%d nodes (%.1f%%) in %.1fs",
                end,
                total,
                end * 100 / total,
                time.perf_counter() - t_start,
            )

        return nodes
