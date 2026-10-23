import base64
import pickle
from typing import Any

from private_gpt.components.readers.nodes import DocumentRootNode, TreeNode
from private_gpt.components.readers.nodes.partial_node import PartialNode


class DocumentRootNodeV2(DocumentRootNode):
    @classmethod
    def version(cls) -> str:
        return "v2"

    def to_tree_serialization(self) -> str:
        """Get the reduced tree serialization.

        This method serializes the tree into a reduced format, keeping only essential
        information needed to reconstruct it.
        """

        def serialization(node: TreeNode) -> dict[str, Any]:
            return {
                "id_": node.id_,
                "type": node.get_type(),
                "version": node.version(),
                "idx": node.idx,
                "abs_idx": node.abs_idx,
                "depth": node.depth,
                "height": node.height,
                "root_id": node.root_id,
                "parent_id": node.parent.id_ if node.parent else None,
                "children_ids": [child.id_ for child in node.children],
                "metadata": {
                    "token_count": node.token_count,
                },
            }

        flatten_tree = [
            serialization(node) for node in self.flatten() if node.id_ != self.id_
        ]
        return base64.b64encode(pickle.dumps(flatten_tree)).decode("utf-8")

    def from_tree_serialization(self, tree_serialization: str) -> None:
        """Create a DocumentRoot from a tree serialization."""
        tree_array: list[dict[str, Any]] = pickle.loads(
            base64.b64decode(tree_serialization.encode("utf-8"))
        )
        tree_dict = {node["id_"]: node for node in tree_array}

        tree_partials: dict[str, TreeNode] = {}
        for node_dict in tree_array:
            tree_partials[node_dict["id_"]] = PartialNode.from_partial_dict(node_dict)

        for partial_node_id, partial_node in tree_partials.items():
            serialized_node = tree_dict[partial_node_id]
            children_ids = serialized_node.get("children_ids", set())
            if children_ids:
                for child_id in children_ids:
                    partial_node.add_child(tree_partials[child_id])

        root_children = [
            tree_partials[child_id]
            for child_id, child in tree_dict.items()
            if child.get("parent_id") == self.id_
        ]
        for child in root_children:
            self.add_child(child)
