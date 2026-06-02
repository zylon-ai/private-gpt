import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.ingestion import arun_transformations
from llama_index.core.schema import BaseNode
from pydantic import Field

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.metadata_helper import MetadataChunk
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.docling.api_clients import (
    BaseDoclingClient,
    DoclingApiOutputModel,
    DoclingClientFactory,
    DoclingConfig,
)
from private_gpt.components.readers.docling.common import (
    DEFAULT_IMAGE_PLACEHOLDER,
    PAGE_PLACEHOLDER,
)
from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER
from private_gpt.settings.settings import (
    DoclingSettings,
    TransformationReadersSettings,
    settings,
)

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class DoclingApiReader(IngestionReader):
    """Docling Reader API.

    Extracts PDF, DOCX, and other document formats into LlamaIndex Documents
    as either Markdown or JSON-serialized Docling native format.

    Args:
        docling_settings (DoclingConfig): Docling settings
    """

    config: DoclingConfig = Field(description="Docling settings")
    client: BaseDoclingClient = Field(description="Docling API client")
    reader_settings: TransformationReadersSettings = Field(
        description="Reader settings"
    )

    def __init__(
        self,
        docling_settings: DoclingSettings,
        reader_settings: TransformationReadersSettings,
        llm_component: LLMComponent | None,
    ) -> None:
        has_image_descriptor_enabled = docling_settings.image_descriptor == "zylon"
        config = DoclingConfig(
            **docling_settings.model_dump(),
            has_image_multimodal_model=has_image_descriptor_enabled
            and (
                any(llm_component.filter(lambda llm, cfg: supports_images(llm, cfg)))
                if llm_component
                else False
            ),
        )
        client = DoclingClientFactory.create(
            config=config,
            async_client=docling_settings.use_async,
        )
        super().__init__(  # type: ignore[call-arg]
            config=config,
            client=client,
            reader_settings=reader_settings,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DoclingApiReader"

    def _page_to_doc(
        self,
        content: str,
        index: int,
        include_page_metadata: bool,
        extra_info: dict[str, Any] | None = None,
    ) -> LIDocument:
        li_doc = LIDocument(
            doc_id=str(uuid.uuid4()),
            text=content,
        )

        # Set extra info if provided
        li_doc.metadata = extra_info or {}
        if include_page_metadata:
            li_doc.metadata[MetadataChunk.PAGE.value] = index + 1  # 1-indexed
            li_doc.excluded_llm_metadata_keys.append(MetadataChunk.PAGE.value)
            li_doc.excluded_embed_metadata_keys.append(MetadataChunk.PAGE.value)

        return li_doc

    def _get_content(
        self,
        conversion_result: DoclingApiOutputModel,
    ) -> list[str]:
        def post_process_content(c: str) -> list[str]:

            result = c

            # Replace API image placeholder with a custom one
            result = result.replace(DEFAULT_IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER)

            # Split pages into contents
            return result.split(PAGE_PLACEHOLDER)

        content = (
            conversion_result.document.md_content
            or conversion_result.document.text_content
            or conversion_result.document.html_content
        )
        if not content:
            raise ValueError("No valid content found in the conversion result")

        return post_process_content(content)

    async def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        notification: NotifyProtocol | None = None,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        """Lazy load file data into LlamaIndex Documents."""
        logger.debug("Starting Docling API parsing of file: %s", file_info.file_name)

        file_name = file_info.file_name or file_info.file_data.name
        file_data = file_info.file_data
        file_bytes = await asyncio.to_thread(file_data.read_bytes)
        pages = file_info.config.get("pages", None)

        try:
            conversion_result = await self.client.convert_from_bytes(
                file_name, file_bytes, to_formats=["md"], pages=pages, **load_kwargs
            )
        except Exception as e:
            raise ValueError(f"Document conversion failed: {e}") from e

        if conversion_result.status not in ["success", "partial_success"]:
            raise ValueError(
                f"Document conversion failed with status: {conversion_result.status}. "
                f"Errors: {conversion_result.errors}"
            )

        contents = self._get_content(conversion_result)
        valid_contents = [content for content in contents if content]
        if not valid_contents:
            raise ValueError("No valid document content found after conversion")

        docs = [
            self._page_to_doc(
                content=content,
                index=idx,
                include_page_metadata=len(valid_contents) > 1,
                extra_info=extra_info,
            )
            for idx, content in enumerate(valid_contents)
        ]

        if debug_mode:
            logger.info(f"Loaded document from {file_name}")
            logger.info(f"Document has {len(docs)} pages")
            if conversion_result.timings:
                logger.debug("Following are the timings for the document conversion:")
                for key, value in conversion_result.timings.items():
                    obj = {
                        "scope": value.scope.value,
                        "count": value.count,
                        "avg": value.avg(),
                        "std": value.std(),
                        "mean": value.mean(),
                    }
                    obj_str = ", ".join([f"{k}: {v}" for k, v in obj.items()])
                    logger.debug(f"{key}: {obj_str}")

        logger.debug(
            "Finished Docling API parsing of file: %s.",
            file_info.file_name,
        )

        if not execute_transformations:
            logger.debug(
                "Skipping transformations for file: %s",
                file_info.file_name,
            )
            for doc in docs:
                yield doc
            return

        logger.debug(
            "Starting Docling API transformations of file: %s",
            file_info.file_name,
        )

        from private_gpt.components.readers.docling.docling_transforms import (
            docling_transformations,
        )

        for transformed_node in await arun_transformations(
            nodes=docs,
            transformations=list(docling_transformations(self.reader_settings)),
        ):
            yield transformed_node

        logger.debug(
            "Finished Docling API parsing and transformations of file: %s",
            file_info.file_name,
        )
