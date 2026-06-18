import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from injector import inject, singleton
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores import FilterCondition, FilterOperator

from private_gpt.artifact_index.artifact_exception import (
    InvalidFileError,
)
from private_gpt.artifact_index.base_artifact_index import (
    ArtifactIndexStatus,
    ExtendIndex,
)
from private_gpt.celery.notify import NotifyProtocol, ProgressStatus, notify_progress
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.fake_progress import (
    calculate_parsing_timing,
    calculate_validation_timing,
)
from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.components.ingest.metadata_helper import MetadataChunk, MetadataKeys
from private_gpt.components.ingest.progress.errors import (
    IngestionLoadErrors,
    IngestionParseErrors,
)
from private_gpt.components.ingest.progress.models import (
    ParseProgressStatus,
    StorageProgressStatus,
    ValidationProgressStatus,
)
from private_gpt.components.ingest.utils import (
    FileInfo,
    convert_unsupported_file,
    convert_unsupported_file_as_fallback,
    get_file_info,
    get_file_name,
    get_filesize,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.readers.docling.docling_api_reader import (
    ExtractionUnsuccessfulError,
)
from private_gpt.components.readers.reader_component import ReaderComponent
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@singleton
class IngestComponent:
    settings: Settings
    node_store_component: NodeStoreComponent
    llm_component: LLMComponent
    embedding_component: EmbeddingComponent
    reader_component: ReaderComponent

    _generate_fake_percentage: bool
    _enable_reuse_generated_nodes_before: bool
    _enable_vision_fallback: bool

    @inject
    def __init__(
        self,
        settings: Settings,
        node_store_component: NodeStoreComponent,
        llm_component: LLMComponent,
        embedding_component: EmbeddingComponent,
        reader_component: ReaderComponent,
    ) -> None:
        self.settings = settings
        self.node_store_component = node_store_component
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.reader_component = reader_component

        self._generate_fake_percentage = settings.data.enable_fake_progress
        self._enable_reuse_generated_nodes_before = (
            settings.data.enable_reuse_generated_nodes_before
        )
        self._enable_vision_fallback = settings.data.enable_vision_fallback

    def load_and_validate_file(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> tuple[FileInfo, list[str], list[str]]:
        # Generate a jitter using the file size between 0.1 and 5
        file_size = get_filesize(file_data)
        interval, jitter = calculate_validation_timing(file_size=file_size)

        with notify_progress(
            notify=notify,
            status_class=ValidationProgressStatus,
            warnings=warnings,
            generate_fake_percentage=self._generate_fake_percentage,
            generate_fake_percentage_interval_ms=int(interval * 1000)
            if interval
            else None,
            generate_fake_percentage_jitter=jitter,
        ) as progress:
            logger.info("Validating file: %s", file_data)
            file_info = self._get_file_info(file_data, file_metadata, progress)
            errors, warnings = self._validate_file(file_info, progress)
            logger.info("Finished validating file: %s", file_data)
            return file_info, errors, warnings

    def _get_file_info(
        self,
        file_data: Path,
        file_metadata: dict[str, Any] | None,
        progress: NotifyProtocol | None = None,
    ) -> FileInfo:
        file_name = get_file_name(file_metadata)
        return get_file_info(file_data, file_name=file_name, progress=progress)

    def _validate_file(
        self,
        file_info: FileInfo,
        progress: NotifyProtocol,
    ) -> tuple[list[str], list[str]]:
        errors, warnings = IngestionHelper.validate_file_info(file_info)
        if errors:
            logger.info("Validation errors: %s", errors)
            raise InvalidFileError(errors=errors, warnings=warnings)
        if warnings:
            logger.info("Validation warnings: %s", warnings)
            progress(percentage=100, warnings=warnings)
        return errors, warnings

    def parse_file_into_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        # Calculate if another file was ingested before
        exists, nodes = self.retrieve_ingested_nodes(artifact, collection, file_info)
        if exists:
            return nodes or []

        # Transform file into nodes
        return self.transform_file_into_nodes(
            artifact=artifact,
            collection=collection,
            file_info=file_info,
            file_metadata=file_metadata,
            notify=notify,
            warnings=warnings,
        )

    def retrieve_ingested_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
    ) -> tuple[bool, list[BaseNode] | None]:
        """Try to find if exists a node with the same hash in the node store."""
        if not file_info.hash:
            return False, None

        filter_dicts = [
            {
                "key": MetadataKeys.FILE_HASH.value,
                "value": file_info.hash,
                "operator": FilterOperator.EQ,
            }
        ]

        # 1. Try to retrieve nodes from same artifact/collection with the same hash
        nodes = self.node_store_component.filtered_nodes(
            collection=collection,
            artifacts=[artifact],
            filter_dicts=filter_dicts,
            limit=1,
            filter_condition=FilterCondition.AND,
        )
        if nodes:
            logger.info("Artifact is already ingested in the node store. Skipping.")
            return True, []

        # 2. Try to retrieve nodes with same hash but different artifact
        if self._enable_reuse_generated_nodes_before:
            nodes = self.node_store_component.filtered_nodes(
                collection=collection,
                artifacts=None,
                filter_dicts=filter_dicts,
            )
            if nodes:
                logger.info(
                    "Found existing nodes with the same hash and "
                    "different artifact in the node store."
                )

                # Group by artifact and collection
                grouped_nodes: dict[str, list[BaseNode]] = {}
                for node in nodes:
                    n_artifact: str = str(
                        node.metadata.get(MetadataKeys.ARTIFACT_ID.value, None)
                    )
                    n_collection: str = str(
                        node.metadata.get(MetadataKeys.COLLECTION.value, None)
                    )
                    key = f"{n_artifact}_{n_collection}"
                    if key not in grouped_nodes:
                        grouped_nodes[key] = []
                    grouped_nodes[key].append(node)

                # Take only one since all artifacts are the same
                chosen_nodes = grouped_nodes.popitem()[1]

                # Update artifact and collection metadata with the new values
                for node in chosen_nodes:
                    node.metadata[MetadataKeys.ARTIFACT_ID.value] = artifact
                    node.metadata[MetadataKeys.COLLECTION.value] = collection
                return True, nodes

        # 3. No nodes found
        return False, None

    def transform_file_into_nodes(
        self,
        artifact: str,
        collection: str,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        """Transform a file into a list of documents.

        This class should be used to transform a file into a list of documents.
        These methods are thread-safe (and multiprocessing-safe).
        """
        interval, jitter = calculate_parsing_timing(
            file_size=file_info.file_size,
            pages=file_info.config.get(MetadataChunk.PAGE.value, 1),
        )
        with notify_progress(
            notify=notify,
            status_class=ParseProgressStatus,
            warnings=warnings,
            generate_fake_percentage=self._generate_fake_percentage,
            generate_fake_percentage_interval_ms=int(interval * 1000)
            if interval
            else None,
            generate_fake_percentage_jitter=jitter,
        ) as notification:
            logger.info("Transforming file into documents: %s", file_info.file_name)

            nodes: list[BaseNode] = []
            try:
                converted_file = convert_unsupported_file(file_info)
                nodes = self._load_data(
                    converted_file,
                    file_metadata,
                    notification=notification,
                    warnings=warnings,
                )
            except ExtractionUnsuccessfulError as e:
                logger.warning(
                    "Extraction unsuccessful for %s: %s", file_info.file_name, e
                )
                try:
                    vision_nodes = self._extract_with_vision_fallback(
                        converted_file,
                        file_metadata,
                        notification=notification,
                        warnings=warnings,
                    )
                except Exception as vision_error:
                    # Vision reader was available but failed during extraction
                    logger.error(
                        "Vision reader fallback failed for %s: %s",
                        file_info.file_name,
                        vision_error,
                        exc_info=True,
                    )
                    raise InvalidFileError(
                        errors=[IngestionParseErrors.PARSING_FAILURE],
                        warnings=warnings,
                    ) from vision_error

                if vision_nodes:
                    nodes = vision_nodes
                else:
                    raise InvalidFileError(
                        errors=[IngestionParseErrors.PARSING_FAILURE],
                        warnings=warnings,
                    ) from e
            except RuntimeError as e:
                raise InvalidFileError(
                    errors=[IngestionParseErrors.PARSING_FAILURE],
                ) from e
            except Exception as e:
                logger.error(f"Error loading file: {e}", exc_info=True)
                converted_file_fallback = convert_unsupported_file_as_fallback(
                    file_info
                )
                if converted_file_fallback:
                    notification(
                        percentage=0,
                        warnings=[IngestionParseErrors.FALLBACK_TO_PDF_TO_TEXT],
                    )
                    nodes = self._load_data(
                        converted_file_fallback,
                        file_metadata,
                        notification=notification,
                        warnings=warnings,
                    )

            if not nodes:
                logger.info("No valid nodes found in the file.")
                raise InvalidFileError(
                    errors=[IngestionLoadErrors.NO_VALID_FILES], warnings=warnings
                )

            max_nodes = self.node_store_component.max_nodes
            if max_nodes and len(nodes) > max_nodes:
                logger.info(
                    "Number of nodes (%d) exceeds the maximum number of nodes (%d)",
                    len(nodes),
                    self.node_store_component.max_nodes,
                )
                raise InvalidFileError(
                    errors=[IngestionParseErrors.PARSING_FAILURE], warnings=warnings
                )

            for document in nodes:
                # Store artifact and collection metadata
                document.metadata[MetadataKeys.ARTIFACT_ID.value] = artifact
                document.metadata[MetadataKeys.COLLECTION.value] = collection

                # Store LLM and Embedding model metadata
                # to know which models were used to ingest the document
                llm_model = self.llm_component.alias
                if llm_model:
                    document.metadata[MetadataKeys.LLM_MODEL.value] = llm_model
                embed_model = self.embedding_component.get_alias()
                if embed_model:
                    document.metadata[MetadataKeys.EMBED_MODEL.value] = embed_model
                document.metadata.update(file_metadata or {})

                # Store current file hash
                document.metadata[MetadataKeys.FILE_HASH.value] = file_info.hash

            IngestionHelper.exclude_metadata(
                nodes=nodes,
                file_metadata=file_metadata,
            )
            logger.info(
                "Finished transforming file into documents: %s", file_info.file_name
            )

        return nodes

    def _load_data(
        self,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        reader_name: str | None = None,
        notification: NotifyProtocol | None = None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        return asyncio.run(
            self._aload_data(
                file_info=file_info,
                file_metadata=file_metadata,
                reader_name=reader_name,
                notification=notification,
                warnings=warnings,
            )
        )

    async def _aload_data(
        self,
        file_info: FileInfo,
        file_metadata: dict[str, Any] | None,
        reader_name: str | None = None,
        notification: NotifyProtocol | None = None,
        warnings: list[str] | None = None,
    ) -> list[BaseNode]:
        if reader_name:
            loader = self.reader_component.get_reader(
                reader_name, file_info.extension or ""
            )
        else:
            loader = self.reader_component.get_reader_by_extension(
                file_info.extension or ""
            )
        nodes: list[BaseNode] = []
        async for node in loader.lazy_load_data(
            file_info,
            extra_info=file_metadata,
            notification=notification,
            warnings=warnings,
        ):
            nodes.append(node)
        return nodes

    def _extract_with_vision_fallback(
        self,
        converted_file: FileInfo,
        file_metadata: dict[str, Any] | None,
        notification: NotifyProtocol,
        warnings: list[str] | None = None,
    ) -> list[BaseNode] | None:
        """Retry extraction of a PDF using the vision reader.

        Returns the extracted nodes, or ``None`` when the fallback does not
        apply (disabled / not a PDF), the vision reader is not available in
        this deployment (logged as a warning), or the vision reader produced
        no usable text (e.g. VLM in mode="none" rasterizing without OCR). If
        the vision reader *is* available but raises during extraction, the
        exception is propagated to the caller.
        """
        if not self._enable_vision_fallback:
            return None

        extension = (converted_file.extension or "").lower()
        if extension != ".pdf":
            return None

        # Availability check: factory registered + VLM instantiable.
        # If not available, degrade gracefully (decision #3).
        try:
            self.reader_component.get_reader("vision", extension)
        except Exception as availability_error:
            logger.warning(
                "Vision reader fallback not available for %s; skipping. Reason: %s",
                converted_file.file_name,
                availability_error,
            )
            return None

        logger.info("Falling back to vision reader for %s", converted_file.file_name)
        vision_nodes = self._load_data(
            converted_file,
            file_metadata,
            reader_name="vision",
            notification=notification,
            warnings=warnings,
        )

        # Guard: a VLM in mode="none" may rasterize pages but return nodes
        # with empty text. Treat "no usable text" as a failed extraction.
        if not vision_nodes or all(
            not node.get_content(metadata_mode=MetadataMode.NONE).strip()
            for node in vision_nodes
        ):
            logger.warning(
                "Vision reader produced no usable text for %s; treating as failure.",
                converted_file.file_name,
            )
            return None

        return vision_nodes

    def load_index(
        self,
        artifact: str,
        collection: str,
        index: BaseIndex[Any],
        index_id: str,
        nodes: list[BaseNode],
        notify: Callable[[ProgressStatus], None] = lambda x: None,
        use_async: bool = True,
        warnings: list[str] | None = None,
    ) -> None:
        """Load the index with the given documents."""
        if not nodes:
            # No nodes to insert
            return

        with notify_progress(
            notify=notify,
            status_class=StorageProgressStatus,
            warnings=warnings,
        ) as notify_publisher:
            logger.info("Loading index %s with %d nodes", index_id, len(nodes))
            max_context_window = self.embedding_component.get_config().context_window
            extended_index = ExtendIndex(
                source=index,
                embed_size=self.settings.vectorstore.embed_dim,
                max_truncate_length=max_context_window * 10,  # 10x context window
            )
            # 1. Delete previous nodes, to avoid duplicates
            self.node_store_component.delete_filtered_nodes(
                collection=collection,
                artifacts=[artifact],
            )

            # 2. Insert nodes
            inserted_nodes: Sequence[BaseNode] = []
            if use_async:
                inserted_nodes = asyncio.run(
                    extended_index.ainsert(nodes, notify=notify_publisher)
                )
            else:
                inserted_nodes = extended_index.insert(nodes, notify=notify_publisher)

            if not inserted_nodes:
                raise InvalidFileError(
                    errors=[IngestionLoadErrors.NO_VALID_NODES], warnings=warnings
                )

            index.summary = ArtifactIndexStatus.POPULATED.value
            index.set_index_id(index_id)
            index.storage_context.persist(persist_dir=local_data_path / collection)
            logger.info("Finished loading index %s with %d nodes", index_id, len(nodes))
