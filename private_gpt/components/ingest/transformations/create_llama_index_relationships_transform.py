"""Simple node parser."""

from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, NodeRelationship, TransformComponent
from llama_index.core.utils import get_tqdm_iterable

from private_gpt.components.readers.nodes.tree_node import TreeNode


class CreateLlamaIndexRelationshipsTransform(TransformComponent):
    """Create LlamaIndex relationships transform.

    Creates relationships between nodes (legacy).
    """

    @classmethod
    def from_defaults(
        cls,
    ) -> "CreateLlamaIndexRelationshipsTransform":
        return cls()

    def __call__(
        self, nodes: Sequence["BaseNode"], **kwargs: Any
    ) -> Sequence["BaseNode"]:
        return self._parse_nodes(nodes, **kwargs)

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        def process_node(
            node: TreeNode, root: TreeNode, prev_node: TreeNode | None
        ) -> TreeNode | None:
            """Process a node to establish relationships."""
            node.relationships[NodeRelationship.SOURCE] = root.as_related_node_info()

            # Update previous relationship
            if (
                prev_node
                and prev_node.source_node
                and node.source_node
                and prev_node.source_node == node.source_node
            ):
                node.relationships[NodeRelationship.PREVIOUS] = (
                    prev_node.as_related_node_info()
                )
                prev_node.relationships[NodeRelationship.NEXT] = (
                    node.as_related_node_info()
                )

            last_processed: TreeNode | None = node
            for child in node.children:
                if isinstance(child, TreeNode):
                    last_processed = process_node(child, root, last_processed)
            return last_processed

        roots = [
            node for node in nodes if isinstance(node, TreeNode) and not node.parent
        ]
        roots_with_progress = get_tqdm_iterable(
            roots, show_progress, "Creating relationships"
        )

        for root in roots_with_progress:
            process_node(root, root, None)

        return list(nodes)
