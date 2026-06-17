import logging
import re
from collections.abc import Iterable

from llama_index.core.schema import TransformComponent

from private_gpt.components.ingest.transformations.combine_tree_transform import (
    CombineTreeTransform,
)
from private_gpt.components.ingest.transformations.convert_image_to_full_slide_transform import (
    ConvertImageToFullSlideTransform,
)
from private_gpt.components.ingest.transformations.create_llama_index_relationships_transform import (
    CreateLlamaIndexRelationshipsTransform,
)
from private_gpt.components.ingest.transformations.describe_image_transform import (
    DescribeImageTransform,
)
from private_gpt.components.ingest.transformations.extract_slide_content_from_image_and_text_transform import (
    ExtractSlideContentFromImageAndTextTransform,
)
from private_gpt.components.ingest.transformations.flatten_tree_nodes_transform import (
    FlattenTreeNodesTransform,
)
from private_gpt.components.ingest.transformations.include_token_count_to_nodes_transform import (
    IncludeTokenCountIntoNodesTransform,
)
from private_gpt.components.ingest.transformations.mark_hidden_nodes_transform import (
    MarkHiddenNodesTransform,
)
from private_gpt.components.ingest.transformations.mark_no_prunable_nodes_transform import (
    MarkNoPrunableNodesTransform,
)
from private_gpt.components.ingest.transformations.markdown_normalization_transform import (
    MarkdownNormalizerTransform,
)
from private_gpt.components.ingest.transformations.markdown_to_tree_transform import (
    MarkdownTreeNodeParser,
)
from private_gpt.components.ingest.transformations.refresh_tree_node_transform import (
    RefreshTreeNodeTransform,
)
from private_gpt.components.ingest.transformations.replace_images_for_placeholder import (
    ReplaceImageByPlaceholder,
)
from private_gpt.components.ingest.transformations.sentence_tree_node_parser import (
    SentenceTreeNodeParser,
)
from private_gpt.components.ingest.transformations.slide_header_checker_transform import (
    AddTitleHeaderTransform,
    ReduceHeaderLevelsTransform,
)
from private_gpt.components.ingest.transformations.vision_docs_transformations import ExtractDocumentContentFromImage
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)


def vision_docs_transformations(
    reader_settings: TransformationReadersSettings, vision_mode: str
) -> Iterable[TransformComponent]:
    # Deduplicate images in text(if apply)
    # yield ImageDeduplicationTransform.from_defaults()
    logger.info(f"PDF transformations with vision_mode: {vision_mode}")

    # For vision docs, we can directly extract the content from the image,
    # as it is likely to be more accurate than OCR text. We can skip
    # the step of converting the image to a full slide and describing it,
    # as we are not working with slides.
    yield ExtractDocumentContentFromImage.from_defaults(
        reader_settings=reader_settings,
    )

    yield MarkdownNormalizerTransform.from_defaults()
    # Merge continuation content into the same page
    # yield MakeContinuationMarkdownTransform.from_defaults()
    # Convert markdown to tree nodes
    yield MarkdownTreeNodeParser.from_defaults(
        include_metadata=True,
    )
    # Create text chunks from the tree nodes
    yield SentenceTreeNodeParser.from_defaults(
        # Include metadata in the nodes
        # generated from the text chunks
        include_metadata=True,
        # We cannot include previous/next relationships as we are not
        # working with a plain list
        include_prev_next_rel=False,
    )
    # Combine all pages into a single document
    yield CombineTreeTransform.from_defaults()
    # Hidden Section nodes that are not real
    yield MarkHiddenNodesTransform.from_defaults(hidden_regex=r"^#\s+Slide\s+\d+$")
    # Mark Section nodes that are empty as non-pruneable
    yield MarkNoPrunableNodesTransform.from_defaults()
    # Flatten the tree nodes
    yield FlattenTreeNodesTransform.from_defaults()
    # Create relationships between nodes (Legacy). Equivalent to the
    # include_prev_next_rel in SentenceTreeNodeParser
    yield CreateLlamaIndexRelationshipsTransform.from_defaults()
    # Include token length as metadata
    yield IncludeTokenCountIntoNodesTransform.from_defaults()
    # Be sure that references are right
    yield RefreshTreeNodeTransform.from_defaults()
