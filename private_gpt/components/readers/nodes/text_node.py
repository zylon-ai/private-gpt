from typing import Union

from llama_index.core.schema import MetadataMode
from pydantic import Field

from private_gpt.components.ingest.metadata_helper import MetadataFlags
from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}{separator}{content}"


class TextNode(TreeNode):
    """Text tree node."""

    text: str = Field(default="", description="Text content of the node.")
    content_type: str = Field(default="text/plain", description="Type of content.")

    text_separator: str = Field(
        default="",
        description="Separator between text fields when converting to string.",
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get object content."""
        hidden = self.metadata.get(MetadataFlags.HIDDEN.value, False)
        if hidden and metadata_mode in [
            TreeMetadataMode.LLM,
            TreeMetadataMode.EMBED,
            TreeMetadataMode.USER,
        ]:
            # Skip the content of self
            return self.text_separator.join(
                child.get_content(metadata_mode) for child in self.children
            )

        content = self.text
        child_content = ""
        if (
            metadata_mode != TreeMetadataMode.NONE
            and metadata_mode != TreeMetadataMode.RAG
        ):
            if self.children:
                child_content = self.text_separator.join(
                    child.get_content(metadata_mode) for child in self.children
                )

        text = self.text_separator.join(filter(None, [content, child_content]))
        if metadata_mode == TreeMetadataMode.EMBED:
            # Remove any image placeholder in EMBED mode
            text = text.replace(IMAGE_PLACEHOLDER, "")

        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        return self.text_template.format(
            content=text,
            separator=self.metadata_separator if metadata_str and content else "",
            metadata_str=metadata_str,
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.text = value

    def prune(
        self,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.LLM,
    ) -> Union["TreeNode", None]:
        """Since we don't support images, we want to discard any placeholder node."""
        no_prunable_node = self.metadata.get(MetadataFlags.NO_PRUNABLE.value, False)
        if no_prunable_node:
            return self
        if self.text.strip() == IMAGE_PLACEHOLDER:
            return None
        return super().prune(metadata_mode)
