import asyncio
import base64
import io
import logging
import subprocess
import tempfile
import uuid
from collections.abc import AsyncIterable
from enum import Enum
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import arun_transformations
from llama_index.core.schema import BaseNode, Document
from PIL import Image

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.progress.errors import IngestionParseErrors
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.vision.vision_transforms import (
    vision_docs_transformations,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)

# Render scale relative to 72 DPI base. 1.5 => ~108 DPI.
# Lower than slide renders: documents are dense text which benefits from
# JPEG at this resolution, staying well within VLM context windows.
_RENDER_SCALE = 1.5


class MetadataChunk(Enum):
    PAGE = "page"


class VisionReader(IngestionReader):
    """Vision-only reader for PDF documents."""

    _reader_settings: TransformationReadersSettings

    def __init__(self, reader_settings: TransformationReadersSettings) -> None:
        super().__init__()
        self._reader_settings = reader_settings

    def _render_pdf_to_images(self, file_path: Path, scale: float) -> list[bytes]:
        """Rasterize every page of the PDF to JPEG bytes via Ghostscript."""
        resolution = int(72 * scale)
        images: list[bytes] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            output_pattern = str(temp_path / "page-%02d.png")

            cmd = [
                "gs",
                "-sDEVICE=png16m",
                f"-o{output_pattern}",
                f"-r{resolution}",
                "-dNOPAUSE",
                "-dBATCH",
                str(file_path),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    stderr_text = (
                        result.stderr.decode("utf-8", errors="replace")
                        if result.stderr
                        else "No error output"
                    )
                    raise RuntimeError(
                        f"Ghostscript failed for {file_path.name}: {stderr_text}"
                    )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Ghostscript timed out for {file_path.name}"
                ) from None

            png_files = sorted(temp_path.glob("page-*.png"))
            for png_file in png_files:
                pil_image = Image.open(png_file)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85, optimize=True)
                images.append(buffer.getvalue())

        logger.info(
            "Rendered %d pages from %s (gs, %ddpi)",
            len(images),
            file_path.name,
            resolution,
        )
        return images

    def _page_to_doc(
        self,
        image_b64: str,
        index: int,
        extra_info: dict[str, Any] | None = None,
    ) -> Document:
        # Empty text on purpose: the markdown is produced downstream by the
        # vision transforms from the page image stored in metadata.
        doc = Document(doc_id=str(uuid.uuid4()), text="")

        doc.metadata = dict(extra_info or {})
        # Key consumed by the vision transforms (kept as "slide_image" to reuse
        # the existing pptx transforms without modification).
        doc.metadata["slide_image"] = image_b64
        doc.metadata[MetadataChunk.PAGE.value] = index + 1

        # Never embed/send the raw base64 image as metadata.
        for key in ("slide_image", MetadataChunk.PAGE.value):
            doc.excluded_llm_metadata_keys.append(key)
            doc.excluded_embed_metadata_keys.append(key)

        return doc

    def _create_docs(
        self,
        page_images: list[bytes],
        extra_info: dict[str, Any] | None,
    ) -> list[Document]:
        docs: list[Document] = []
        for index, image_bytes in enumerate(page_images):
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            docs.append(
                self._page_to_doc(
                    image_b64=image_b64,
                    index=index,
                    extra_info=extra_info,
                )
            )
        return docs

    async def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        notification: NotifyProtocol | None = None,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterable[BaseNode]:

        # Resolve the vision mode (deep / lite / none) the same way as pptx.
        vision_mode: str = "none"
        if self._reader_settings and self._reader_settings.vision.is_enabled:
            from private_gpt.components.llm.llm_component import LLMComponent

            llm_component = get_global_injector().get(LLMComponent)
            have_multimodal_model = (
                any(llm_component.filter(lambda llm, cfg: supports_images(llm, cfg)))
                if llm_component
                else False
            )
            vision_mode = self._reader_settings.vision.get_vision_mode(
                have_multimodal_model
            )

        # Rasterize the PDF (blocking / CPU-bound -> thread).
        page_images = await asyncio.to_thread(
            self._render_pdf_to_images,
            file_info.file_data.absolute(),
            _RENDER_SCALE,
        )
        docs = self._create_docs(page_images=page_images, extra_info=extra_info)

        if self._reader_settings.vision.is_enabled and notification:
            notification(
                percentage=0,
                warnings=[IngestionParseErrors.USING_VLM_FOR_EXTRACTION],
            )

        logger.info(f"Created {len(docs)} documents from {file_info.file_name}")

        if not execute_transformations:
            logger.debug("Skipping transformations for file: %s", file_info.file_name)
            for node in docs:
                yield node
            return

        logger.debug(
            "Starting PDF vision transformations of file: %s", file_info.file_name
        )

        for transformed_node in await arun_transformations(
            nodes=docs,
            transformations=list(
                vision_docs_transformations(
                    reader_settings=self._reader_settings, vision_mode=vision_mode
                )
            ),
        ):
            yield transformed_node

        logger.debug("Finished PDF vision parsing of file: %s", file_info.file_name)
