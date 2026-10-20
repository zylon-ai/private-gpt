import re

from llama_index.core.schema import MetadataMode, TextNode

from private_gpt.components.ingest.transformations.describe_image_transform import (
    DescribeImageTransform,
)
from private_gpt.components.ingest.transformations.replace_images_for_placeholder import (
    ReplaceImageByPlaceholder,
)
from private_gpt.settings.settings import TransformationReadersSettings

_EMBEDDED_IMAGE = re.compile(r"!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]+\)")


async def resolve_document_images(
    text: str,
    image_description: TransformationReadersSettings | None = None,
) -> str:
    """Turn the base64 images embedded in a converted document into text.

    Readers that embed images (e.g. Docling with ``image_mode: embedded``) leave
    them in the markdown as data URIs. Chat conversion skips the ingestion
    transformations, so without this they would reach the prompt as raw base64.

    With ``image_description`` each image is described by the first model that
    supports images. Whatever is left (description disabled, no such model, or
    a failure) is replaced by a placeholder.
    """
    if not _EMBEDDED_IMAGE.search(text):
        return text

    nodes = [TextNode(text=text)]
    if image_description is not None:
        nodes = list(
            await DescribeImageTransform.from_defaults(
                reader_settings=image_description
            ).acall(nodes)
        )
    nodes = list(ReplaceImageByPlaceholder.from_defaults()(nodes))
    return nodes[0].get_content(MetadataMode.NONE)
