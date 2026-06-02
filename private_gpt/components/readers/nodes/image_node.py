from llama_index.core.schema import DEFAULT_TEXT_NODE_TMPL
from pydantic import Field

from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

IMAGE_PLACEHOLDER = "[Image not available]"
IMAGE_NOT_PROCESSABLE = "[Image not processable]"


class ImageNode(TreeNode):
    """Chunk tree node."""

    alt_text: str = Field(default="", description="Alt text for the image.")
    content_type: str = Field(default="text/plain", description="Type of content.")
    image: str = Field(default="", description="Image b64 content of the node.")
    description: str | None = Field(
        default=None,
        description="Description of the image, if any.",
    )

    text_seperator: str = Field(
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
        # TODO: Disable image embeddings until we have a way to handle them
        if metadata_mode == TreeMetadataMode.EMBED:
            return ""

        if (
            metadata_mode == TreeMetadataMode.ALL
            or metadata_mode == TreeMetadataMode.LLM
            or metadata_mode == TreeMetadataMode.USER
        ):
            content = (
                f"![{self.alt_text}]({self.image})" if self.image else self.alt_text
            )
        else:
            content = self.description if self.description else self.alt_text

        if metadata_mode == TreeMetadataMode.LLM and (
            content in (IMAGE_PLACEHOLDER, IMAGE_NOT_PROCESSABLE)
        ):
            content = ""

        child_content = ""
        if (
            metadata_mode != TreeMetadataMode.NONE
            and metadata_mode != TreeMetadataMode.RAG
        ):
            if self.children:
                child_content = self.text_seperator.join(
                    child.get_content(metadata_mode) for child in self.children
                )

        text = self.text_seperator.join(filter(None, [content, child_content]))
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        return self.text_template.format(
            content=text,
            separator=self.metadata_separator if metadata_str and content else "",
            metadata_str=metadata_str,
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.alt_text = value
