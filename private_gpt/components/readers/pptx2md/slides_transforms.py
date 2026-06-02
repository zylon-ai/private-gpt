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
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)


def slides_transformations(
    reader_settings: TransformationReadersSettings, vision_mode: str
) -> Iterable[TransformComponent]:
    # Deduplicate images in text(if apply)
    # yield ImageDeduplicationTransform.from_defaults()
    logger.info(f"Slides transformations with vision_mode: {vision_mode}")
    if vision_mode == "deep":
        # Convert images to the full slide, marking the focus of each image
        yield ConvertImageToFullSlideTransform.from_defaults(
            reader_settings=reader_settings,
        )
        # Create a representation of images in text (if apply)
        yield DescribeImageTransform.from_defaults(
            reader_settings=reader_settings,
        )
    elif vision_mode == "lite":
        # For vision mode, extract complete slide content from:
        # 1) text parsed by pptx2md and 2) full slide image.
        yield ExtractSlideContentFromImageAndTextTransform.from_defaults(
            reader_settings=reader_settings,
        )
    else:
        yield ReplaceImageByPlaceholder.from_defaults()

    # Add title to each slide
    # To do that, we need to guarantee that header levels are below
    has_h1_predicate = lambda content: bool(  # noqa: E731
        re.search(r"^#\s+", content, re.MULTILINE)
    )
    yield ReduceHeaderLevelsTransform.from_defaults(
        reduce_by=1, predicate=has_h1_predicate
    )
    yield AddTitleHeaderTransform.from_defaults(default_title="Slide")
    # Remove header and footer
    # yield RemoveHeaderAndFooterTransform.from_defaults()
    # Normalize markdown indentation
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
