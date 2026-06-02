from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class IncludeTokenCountIntoNodesTransform(TransformComponent):
    """Include token length in the nodes."""

    tokenizer: TokenizerFn | None

    @classmethod
    def from_defaults(
        cls, tokenizer: TokenizerFn | None = None
    ) -> "IncludeTokenCountIntoNodesTransform":
        return cls(tokenizer=tokenizer)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        if not self.tokenizer:
            return nodes

        for node in nodes:
            content = (
                node.get_content(TreeMetadataMode.RAG)
                if isinstance(node, TreeNode)
                else node.get_content(MetadataMode.NONE)
            )
            if not content:
                # Skip to avoid to save a token count of emtpy
                continue
            node.metadata[MetadataNode.TOKEN_COUNT.value] = len(self.tokenizer(content))
            node.excluded_llm_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)
            node.excluded_embed_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)
        return nodes
