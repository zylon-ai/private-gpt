from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, TransformComponent

from private_gpt.components.readers.nodes.tree_node import TreeNode
from private_gpt.components.readers.nodes.utils import combine_trees


class CombineTreeTransform(TransformComponent):
    """Combine all tree nodes into a single."""

    @classmethod
    def from_defaults(
        cls,
    ) -> "CombineTreeTransform":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        if len(nodes) == 1:
            return nodes

        tree_nodes = [
            tree_node for tree_node in nodes if isinstance(tree_node, TreeNode)
        ]
        return [combine_trees(tree_nodes[0], *(tree_nodes[1:]))]
