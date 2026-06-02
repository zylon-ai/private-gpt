import builtins
import json
from hashlib import sha256
from typing import Any, Self

from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class DocumentRootNode(TreeNode):
    """Root node representing the document."""

    @property
    def token_count(self) -> int:
        return sum([child.token_count for child in self.children or []])

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get the content of the document as a string."""
        if (
            metadata_mode == TreeMetadataMode.NONE
            or metadata_mode == TreeMetadataMode.RAG
        ):
            return ""

        return "".join(
            [child.get_content(metadata_mode) for child in self.children or []]
        )

    def set_content(self, value: Any) -> None:
        """Set the content of the node."""
        raise NotImplementedError("Cannot set content for DocumentRoot node.")

    @property
    def hash(self) -> str:
        doc_identity = f"{self.id_}-{self.get_type()}"
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    def model_dump(
        self,
        *,
        include_parent: bool = False,
        include_children: bool = False,
        include_tree: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize the node into a dictionary.

        Exist a memory check to ensure that the serialization is not too large,
        waiting to be serialized until
        the memory is available.
        """
        obj = super().model_dump(
            include_parent=include_parent, include_children=include_children, **kwargs
        )
        # Add tree serialization
        if include_tree:
            tree = self.to_tree_serialization()
            obj["tree"] = tree
        return obj

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        """This method serialize tree into a reduced format.

        Exist a memory check to ensure that the deserialization is not too large,
        waiting to be serialized until the memory is available.
        """
        obj = super().from_dict(data, **kwargs)
        if "tree" in data:
            obj.from_tree_serialization(data["tree"])
        return obj

    @classmethod
    def clone_tree(
        cls,
        node: "TreeNode",
        **kwargs: Any,
    ) -> "TreeNode":
        """Clones the tree structure starting from the given node."""
        return cls.from_dict(
            data=node.dict(
                include_parent=False,
                include_children=False,
                include_tree=False,
                **kwargs,
            )
        )

    def to_tree_serialization(self) -> str:
        """Get the reduced tree serialization.

        This method serializes the tree into a reduced format, keeping only essential
        information needed to reconstruct it.
        """
        tree = [PartialNode.from_node(child).dict() for child in self.children]
        return json.dumps(tree)

    def from_tree_serialization(self, tree_serialization: str) -> None:
        """Create a DocumentRoot from a tree serialization."""
        tree = json.loads(tree_serialization)
        for node in tree:
            deserialized = PartialNode.from_partial_dict(node)
            if deserialized:
                self.add_child(deserialized, update_references=False)

    @staticmethod
    def load_subtree(
        root: TreeNode, flat_nodes: list[TreeNode], verbose: bool = False
    ) -> TreeNode:
        """Loads full node data into a tree containing partial nodes.

        Args:
            root: Root node containing the complete tree structure with partial nodes
            flat_nodes: List of full node data to be loaded into the tree
            verbose: Whether to print warnings about missing nodes

        Returns:
            TreeNode: Root of the processed tree with full nodes
        """
        # Create lookup dictionary for full nodes - O(n)
        node_lookup = {n.id_: n for n in flat_nodes}

        def replace_node(partial: TreeNode) -> TreeNode:
            """Replaces a partial node with its full version."""
            full_node = node_lookup.get(partial.id_, partial)

            # Keep existing children but update their parent reference
            full_node.children = partial.children
            for child in full_node.children:
                child.parent = full_node

            return full_node

        # Process the tree in a single traversal - O(m) where m is tree size
        def process_subtree(node: TreeNode) -> TreeNode:
            # Replace current node if it's partial
            if isinstance(node, PartialNode):
                node = replace_node(node)

            # Process all children
            node.children = [process_subtree(child) for child in node.children]
            return node

        # Process the entire tree starting from root
        root = process_subtree(root)

        if verbose:
            # Verify all nodes are present
            final_nodes = list(root.flatten())
            final_ids = {
                node.id_ for node in final_nodes if not node.isinstance(PartialNode)
            }
            missing_nodes = [node for node in flat_nodes if node.id_ not in final_ids]
            if missing_nodes:
                print(
                    f"Warning: {len(missing_nodes)} nodes are missing in the final tree:"
                )
                for node in missing_nodes:
                    print(
                        f"Missing node: {node.id_}, Parent: {node.parent.id_ if node.parent else 'None'}"
                    )

        return root
