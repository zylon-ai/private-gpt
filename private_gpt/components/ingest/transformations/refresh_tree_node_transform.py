from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, TransformComponent

from private_gpt.components.readers.nodes import DocumentRootNode


class RefreshTreeNodeTransform(TransformComponent):
    """Refresh tree node transform."""

    @classmethod
    def from_defaults(cls) -> "RefreshTreeNodeTransform":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Refresh tree node transform in one shot.

        Args:
            nodes (Sequence[BaseNode]): The nodes to transform.
            kwargs (Any): Additional arguments.

        Returns:
            Sequence[BaseNode]: The transformed nodes.
        """
        root_nodes = [node for node in nodes if isinstance(node, DocumentRootNode)]
        for root_node in root_nodes:
            root_node.update_references()

        return nodes
