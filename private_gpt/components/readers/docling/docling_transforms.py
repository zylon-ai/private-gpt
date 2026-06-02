from collections.abc import Iterable

from llama_index.core.schema import TransformComponent

from private_gpt.components.ingest.transformations.combine_tree_transform import (
    CombineTreeTransform,
)
from private_gpt.components.ingest.transformations.create_llama_index_relationships_transform import (
    CreateLlamaIndexRelationshipsTransform,
)
from private_gpt.components.ingest.transformations.describe_image_transform import (
    DescribeImageTransform,
)
from private_gpt.components.ingest.transformations.flatten_tree_nodes_transform import (
    FlattenTreeNodesTransform,
)
from private_gpt.components.ingest.transformations.include_token_count_to_nodes_transform import (
    IncludeTokenCountIntoNodesTransform,
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
from private_gpt.settings.settings import TransformationReadersSettings


def docling_transformations(
    reader_settings: TransformationReadersSettings,
) -> Iterable[TransformComponent]:
    # Deduplicate images in text(if apply)
    # yield ImageDeduplicationTransform.from_defaults()
    # Create a representation of images in text (if apply)
    if reader_settings.vision.is_enabled:
        yield DescribeImageTransform.from_defaults(
            reader_settings=reader_settings,
        )
    else:
        yield ReplaceImageByPlaceholder.from_defaults()
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
    # Flatten the tree nodes
    yield FlattenTreeNodesTransform.from_defaults()
    # Create relationships between nodes (Legacy). Equivalent to the
    # include_prev_next_rel in SentenceTreeNodeParser
    yield CreateLlamaIndexRelationshipsTransform.from_defaults()
    # Include token length as metadata
    yield IncludeTokenCountIntoNodesTransform.from_defaults()
    # Be sure that references are right
    yield RefreshTreeNodeTransform.from_defaults()
