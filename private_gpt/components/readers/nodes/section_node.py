from llama_index.core.schema import MetadataMode

from private_gpt.components.ingest.metadata_helper import MetadataFlags
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class SectionNode(TextNode):
    """Section node."""

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get object content."""
        if (
            metadata_mode == TreeMetadataMode.NONE
            or metadata_mode == TreeMetadataMode.RAG
        ):
            return self.text

        return super().get_content_internal(metadata_mode)

    def prune(
        self,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.LLM,
    ) -> TreeNode | None:
        """Reduce node."""
        no_prunable_node = self.metadata.get(MetadataFlags.NO_PRUNABLE.value, False)
        if no_prunable_node:
            return self

        if not self.children:
            # If the node is a section node, and it has no children, skip it
            return None

        if not super().prune(metadata_mode):
            return None

        # If the content in the children is not empty, keep the node
        content = self.get_content(metadata_mode).strip()
        content = content.replace(self.text.strip(), "").strip()
        if content:
            return self

        # If the node is a section node, and it has no children with content, skip it
        return None
