"""Simple node parser."""
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import TransformComponent

from private_gpt.components.readers.nodes.tree_node import TreeNode

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode

DEFAULT_WINDOW_SIZE = 3
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


class FlattenTreeNodesTransform(TransformComponent):
    """Sentence node parser.

    Splits a document into Nodes, with each node being a sentence.

    Args:
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    @classmethod
    def from_defaults(
        cls,
    ) -> "FlattenTreeNodesTransform":
        return cls()

    def __call__(
        self, nodes: Sequence["BaseNode"], **kwargs: Any
    ) -> Sequence["BaseNode"]:
        return self.flatten_tree_nodes(nodes)

    def flatten_tree_nodes(self, nodes: Sequence["BaseNode"]) -> Sequence["BaseNode"]:
        """Flatten tree nodes."""
        flattened_nodes: list[BaseNode] = []
        for node in nodes:
            if isinstance(node, TreeNode):
                flattened_nodes.extend(node.flatten())
            else:
                flattened_nodes.append(node)
        return flattened_nodes
