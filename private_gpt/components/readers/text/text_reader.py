import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import arun_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent

from private_gpt.components.ingest.transformations.combine_tree_transform import (
    CombineTreeTransform,
)
from private_gpt.components.ingest.transformations.create_llama_index_relationships_transform import (
    CreateLlamaIndexRelationshipsTransform,
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
from private_gpt.components.ingest.transformations.sentence_tree_node_parser import (
    SentenceTreeNodeParser,
)
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.settings.settings import settings

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


SUPPORTED_ENCODINGS = [
    "utf-8",
    "latin-1",
    "ascii",
    "cp1252",
    "ISO-8859-1",
    "windows-1252",
]


class TextReader(IngestionReader):
    def _process_content(self, content: str) -> str:
        """Process the content of the document.

        This method can be overridden to apply custom processing.
        """
        # Default implementation does nothing, just returns the content
        return content

    def lazy_document_load(
        self,
        file_path: Path,
        encoding: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Iterator[BaseNode]:
        encoding = encoding.lower() if encoding else None
        encodings = SUPPORTED_ENCODINGS.copy()

        if encoding and encoding.lower() not in [
            e.lower() for e in SUPPORTED_ENCODINGS
        ]:
            if encoding in encodings:
                encodings.remove(encoding)
            encodings.insert(0, encoding)

        for enc in encodings:
            try:
                with open(file_path, encoding=enc) as file:
                    content = file.read()

                # Convert content to utf-8
                content = content.encode("utf-8").decode("utf-8")

                # Process content
                content = self._process_content(content)

                # Create and yield the document
                yield Document(
                    text=content,
                    extra_info=extra_info if extra_info is not None else {},
                )
                return
            except UnicodeDecodeError:
                continue  # Try next encoding

    async def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        del args, load_kwargs
        reader_name = self.__class__.__name__
        logger.debug(
            "Starting %s parsing of file: %s", reader_name, file_info.file_name
        )
        documents = await asyncio.to_thread(
            lambda: list(
                self.lazy_document_load(
                    file_path=file_info.file_data,
                    encoding=file_info.encoding,
                    extra_info=extra_info,
                )
            )
        )
        logger.debug(
            "Finished %s parsing of file: %s.",
            reader_name,
            file_info.file_name,
        )
        if not execute_transformations:
            logger.debug(
                "Skipping transformations for file: %s",
                file_info.file_name,
            )
            for document in documents:
                yield document
            return

        logger.debug(
            "Starting %s transformations of file: %s",
            reader_name,
            file_info.file_name,
        )
        transformed_nodes = await arun_transformations(
            documents,
            list(self._tranformations()),
        )
        for node in transformed_nodes:
            yield node
        logger.debug(
            "Finished %s parsing and transformations of file: %s",
            reader_name,
            file_info.file_name,
        )

    def _tranformations(self) -> Iterable[TransformComponent]:
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
