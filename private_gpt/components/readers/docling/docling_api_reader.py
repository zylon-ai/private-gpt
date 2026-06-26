import asyncio
import logging
import random
import uuid
from collections.abc import AsyncIterator
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.ingestion import arun_transformations
from llama_index.core.schema import BaseNode
from pydantic import Field

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.metadata_helper import MetadataChunk
from private_gpt.components.ingest.utils import FileInfo, extract_pdf_page_range
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

GLYPH_UNKNOWN_MARKER = "glyph<UNKNOWN>"
_MAX_SAMPLED_PAGES = 10


class ExtractionUnsuccessfulError(Exception):
    """Raised when a document is converted but the extracted text is unusable.

    (e.g. mostly unmapped glyphs from CID fonts without a ToUnicode map).
    """


def _glyph_unknown_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count(GLYPH_UNKNOWN_MARKER) * len(GLYPH_UNKNOWN_MARKER) / len(text)


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

    async def _do_convert(
        self,
        file_name: str,
        file_bytes: bytes,
        extra_info: dict[str, Any] | None,
        page_offset: int = 0,
        pages: int | None = None,
        **load_kwargs: Any,
    ) -> tuple[list[LIDocument], dict[str, Any]]:
        """Run one Docling conversion call and return (docs, timings).

        *page_offset* shifts page-metadata numbers so that a sub-PDF extracted
        from a larger document gets correct absolute page numbers.
        *pages* is a hint to cap the page range at (1, pages) on v1 API.
        """
        try:
            result = await self.client.convert_from_bytes(
                file_name, file_bytes, to_formats=["md"],
                pages=pages, **load_kwargs
            )
        except Exception as e:
            raise ValueError(f"Document conversion failed: {e}") from e

        if result.status not in ["success", "partial_success"]:
            raise ValueError(
                f"Document conversion failed with status: {result.status}. "
                f"Errors: {result.errors}"
            )

        contents = self._get_content(result)
        valid_contents = [c for c in contents if c]
        if not valid_contents:
            raise ValueError("No valid document content found after conversion")
        if self._is_extraction_unsuccessful(valid_contents):
            raise ExtractionUnsuccessfulError(
                f"Document extraction unsuccessful for '{file_name}': unmapped-glyph "
                f"ratio exceeded threshold ({self.config.failure_threshold})."
            )

        include_page_meta = page_offset > 0 or len(valid_contents) > 1
        docs = [
            self._page_to_doc(
                content=content,
                index=page_offset + idx,
                include_page_metadata=include_page_meta,
                extra_info=extra_info,
            )
            for idx, content in enumerate(valid_contents)
        ]
        return docs, result.timings

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
        file_bytes = await asyncio.to_thread(file_info.file_data.read_bytes)

        total_pages: int | None = file_info.config.get("pages")
        chunk_size = settings().data.limits.chunk_size_pages

        timings: dict[str, Any] = {}

        if total_pages is not None and total_pages > chunk_size:
            chunks = [
                (start, min(start + chunk_size - 1, total_pages))
                for start in range(1, total_pages + 1, chunk_size)
            ]
            logger.debug(
                "Splitting %d-page document into %d chunks of %d pages: %s",
                total_pages, len(chunks), chunk_size, file_name,
            )
            docs: list[LIDocument] = []
            for i, (start, end) in enumerate(chunks):
                chunk_bytes = await asyncio.to_thread(
                    extract_pdf_page_range, file_bytes, start, end
                )
                chunk_docs, _ = await self._do_convert(
                    file_name=file_name,
                    file_bytes=chunk_bytes,
                    extra_info=extra_info,
                    page_offset=start - 1,
                    **load_kwargs,
                )
                docs.extend(chunk_docs)
                if notification:
                    notification(percentage=(i + 1) / len(chunks) * 100)
        else:
            docs, timings = await self._do_convert(
                file_name=file_name,
                file_bytes=file_bytes,
                extra_info=extra_info,
                pages=total_pages,
                **load_kwargs,
            )

        if debug_mode:
            logger.info(f"Loaded document from {file_name}")
            logger.info(f"Document has {len(docs)} pages")
            if timings:
                logger.debug("Following are the timings for the document conversion:")
                for key, value in timings.items():
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

    def _is_extraction_unsuccessful(self, contents: list[str]) -> bool:
        """Detect when Docling failed to extract meaningful text.

        Sample up to ``_MAX_SAMPLED_PAGES`` pages at random and evaluate the
        unmapped-glyph ratio over them. A broken extraction (e.g. CID fonts
        without a ToUnicode map) typically affects the whole document, so a
        random sample estimates it well while bounding work on large docs.
        """
        if not contents:
            return False

        sample_size = min(_MAX_SAMPLED_PAGES, len(contents))
        sampled_indices = random.sample(range(len(contents)), sample_size)
        sampled_text = "".join(contents[i] for i in sampled_indices)

        return _glyph_unknown_ratio(sampled_text) > self.config.failure_threshold
