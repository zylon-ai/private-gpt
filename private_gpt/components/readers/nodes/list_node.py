from pydantic import Field

from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class ListNode(TextNode):
    """List node."""

    num_items: int = Field(
        default=0,
        description="Number of items in the list.",
    )

    def add_child(self, child: TreeNode, update_references: bool = False) -> None:
        """Add a child to the node."""
        super().add_child(child)
        self.num_items += 1

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get object content."""
        if (
            metadata_mode == TreeMetadataMode.NONE
            or metadata_mode == TreeMetadataMode.RAG
        ):
            return ""

        return super().get_content_internal(metadata_mode)


class ListItemNode(TextNode):
    """List Item node."""

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get object content."""
        if metadata_mode == TreeMetadataMode.NONE:
            return self.text

        return super().get_content_internal(metadata_mode)
