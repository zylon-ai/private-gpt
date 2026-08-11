import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.schema import BaseNode
from pydantic import ConfigDict, Field

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.metadata_helper import MetadataChunk
from private_gpt.components.ingest.pdf_page_split import (
    extract_pdf_pages_bytes,
    group_consecutive_pages,
)
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.factory import ReaderFactoryRegistry
from private_gpt.components.readers.pdf_inspector.pdf_inspector_reader import (
    PdfInspectorFallbackError,
)
from private_gpt.components.readers.registry import ReaderRegistry
from private_gpt.settings.settings import (
    PdfInspectorSettings,
    TransformationReadersSettings,
    settings,
)

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class HybridPdfReader(IngestionReader):
    """Splits a PDF into OCR / non-OCR page groups and reassembles them.

    Uses pdf-inspector's per-page classification to extract non-OCR pages
    locally and only route the pages that actually need OCR to a heavier
    reader (e.g. docling, vision), instead of reprocessing the whole
    document. Raises ``PdfInspectorFallbackError`` whenever the hybrid
    split isn't worth it (or fails), so the caller falls back to the next
    full-document reader in the configured chain.
    """

    config: PdfInspectorSettings = Field(description="PDF Inspector settings")
    reader_settings: TransformationReadersSettings = Field(
        description="Reader settings used to run the OCR-reader transformations"
    )
    reader_registry: ReaderRegistry = Field(description="Reader chain registry")
    factory_registry: ReaderFactoryRegistry = Field(description="Reader factories")
    own_reader_name: str = Field(
        default="pdf-inspector-hybrid",
        description="Name this reader is registered under, used to resolve "
        "the 'auto' OCR reader as the next one in the chain",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        pdf_inspector_settings: PdfInspectorSettings,
        reader_settings: TransformationReadersSettings,
        reader_registry: ReaderRegistry,
        factory_registry: ReaderFactoryRegistry,
        own_reader_name: str = "pdf-inspector-hybrid",
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            config=pdf_inspector_settings,
            reader_settings=reader_settings,
            reader_registry=reader_registry,
            factory_registry=factory_registry,
            own_reader_name=own_reader_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HybridPdfReader"

    def _resolve_ocr_reader_name(self, extension: str) -> str:
        if self.config.hybrid_ocr_reader != "auto":
            return self.config.hybrid_ocr_reader

        names = self.reader_registry.get_reader_names(extension)
        try:
            index = names.index(self.own_reader_name)
        except ValueError:
            raise PdfInspectorFallbackError(
                f"'{self.own_reader_name}' is not registered for '{extension}'; "
                "cannot resolve 'auto' OCR reader."
            ) from None
        if index + 1 >= len(names):
            raise PdfInspectorFallbackError(
                f"No reader configured after '{self.own_reader_name}' for "
                f"'{extension}' to handle OCR pages."
            )
        return names[index + 1]

    def _page_to_doc(
        self,
        content: str,
        page: int,
        include_page_metadata: bool,
        extra_info: dict[str, Any] | None = None,
    ) -> LIDocument:
        li_doc = LIDocument(text=content)
        li_doc.metadata = extra_info or {}
        if include_page_metadata:
            li_doc.metadata[MetadataChunk.PAGE.value] = page + 1  # 1-indexed
            li_doc.excluded_llm_metadata_keys.append(MetadataChunk.PAGE.value)
            li_doc.excluded_embed_metadata_keys.append(MetadataChunk.PAGE.value)
        return li_doc

    async def _extract_ocr_group_content(
        self,
        file_path: Path,
        extension: str,
        start: int,
        end: int,
    ) -> list[str]:
        """Run the OCR reader over pages [start, end] and return their content."""
        ocr_reader_name = self._resolve_ocr_reader_name(extension)
        try:
            sub_reader = self.factory_registry.get_factory(
                ocr_reader_name
            ).create_reader(extension)
        except Exception as e:
            raise PdfInspectorFallbackError(
                f"Could not create OCR reader '{ocr_reader_name}': {e}"
            ) from e

        group_bytes = extract_pdf_pages_bytes(file_path, start, end)

        with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as tmp:
            tmp.write(group_bytes)
            tmp.flush()
            tmp_file_info = FileInfo(
                file_name=f"{file_path.stem}_pages_{start}_{end}{extension}",
                extension=extension,
                file_data=Path(tmp.name),
            )

            try:
                docs = [
                    node
                    async for node in sub_reader.lazy_load_data(
                        tmp_file_info, execute_transformations=False
                    )
                ]
            except Exception as e:
                raise PdfInspectorFallbackError(
                    f"OCR reader '{ocr_reader_name}' failed on pages "
                    f"[{start}, {end}]: {e}"
                ) from e

        expected_pages = end - start + 1
        if len(docs) != expected_pages:
            raise PdfInspectorFallbackError(
                f"OCR reader '{ocr_reader_name}' returned {len(docs)} pages "
                f"for group [{start}, {end}], expected {expected_pages}."
            )

        return [doc.get_content() for doc in docs]

    async def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        notification: NotifyProtocol | None = None,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        del notification, args, load_kwargs
        file_path = file_info.file_data
        extension = file_info.extension or ".pdf"

        try:
            import pdf_inspector  # type: ignore[import-not-found]

            logger.debug(
                "Starting hybrid pdf-inspector split of file: %s", file_path
            )
            result = pdf_inspector.extract_pages_markdown(str(file_path))
        except Exception as e:
            raise PdfInspectorFallbackError(
                f"pdf-inspector failed to process '{file_path}'."
            ) from e

        pages = result.pages
        page_count = len(pages)
        if page_count == 0:
            raise PdfInspectorFallbackError(f"'{file_path}' has no pages.")

        ocr_pages = sorted({p - 1 for p in result.pages_needing_ocr})
        ratio = len(ocr_pages) / page_count

        if ratio > self.config.hybrid_ocr_page_threshold:
            raise PdfInspectorFallbackError(
                f"'{file_path}' has {ratio:.2%} of pages needing OCR, above "
                f"threshold ({self.config.hybrid_ocr_page_threshold:.2%})."
            )

        groups = group_consecutive_pages(ocr_pages)
        if len(groups) > self.config.max_ocr_groups:
            raise PdfInspectorFallbackError(
                f"'{file_path}' would require {len(groups)} OCR page groups, "
                f"above the configured limit ({self.config.max_ocr_groups})."
            )

        page_content: dict[int, str] = {
            page.page: page.markdown for page in pages if not page.needs_ocr
        }
        for start, end in groups:
            ocr_content = await self._extract_ocr_group_content(
                file_path, extension, start, end
            )
            for offset, content in enumerate(ocr_content):
                page_content[start + offset] = content

        docs = [
            self._page_to_doc(
                content=page_content.get(page_index, ""),
                page=page_index,
                include_page_metadata=page_count > 1,
                extra_info=extra_info,
            )
            for page_index in range(page_count)
        ]

        logger.debug(
            "Finished hybrid pdf-inspector split of file: %s (%d OCR pages "
            "in %d group(s) of %d total pages)",
            file_path,
            len(ocr_pages),
            len(groups),
            page_count,
        )

        if not execute_transformations:
            for doc in docs:
                yield doc
            return

        from private_gpt.components.readers.docling.docling_transforms import (
            docling_transformations,
        )

        transformed_nodes = await self._run_transformations_with_timing(
            docs,
            docling_transformations(self.reader_settings),
            file_info.file_name,
        )
        for transformed_node in transformed_nodes:
            yield transformed_node
