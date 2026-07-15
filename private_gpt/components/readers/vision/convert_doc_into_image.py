import asyncio
import base64
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterable
from enum import Enum
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import arun_transformations
from llama_index.core.schema import BaseNode, Document
from pptx2md import convert  # ty:ignore[unresolved-import]

from private_gpt.celery.notify import NotifyProtocol
from private_gpt.components.ingest.progress.errors import IngestionParseErrors
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.pptx2md.convert_slide_into_image import (
    ExportedImages,
    PPTXSlideToImageDeepTransform,
    PPTXSlideToImageTransform,
)
from private_gpt.components.readers.pptx2md.slides_transforms import (
    slides_transformations,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)


class MetadataChunk(Enum):
    PAGE = "page"


class VisionReader(IngestionReader):
    _reader_settings: TransformationReadersSettings

    def __init__(self, reader_settings: TransformationReadersSettings) -> None:
        super().__init__()
        self._reader_settings = reader_settings

    def _get_image_mime_type(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
            ".bmp": "image/bmp",
        }
        return mime_types.get(extension, "image/png")

    def _read_file_as_base64(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            content = f.read()
        return base64.b64encode(content).decode("utf-8")

    def _replace_images_with_base64(self, markdown_content: str, temp_dir: Path) -> str:
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        def replace_image(match: re.Match[str]) -> str:
            alt_text = match.group(1)
            image_path_str = match.group(2)

            if image_path_str.startswith("/img/"):
                image_path_str = image_path_str[1:]

            if (
                image_path_str.startswith("img/")
                or not Path(image_path_str).is_absolute()
            ):
                image_path = temp_dir / image_path_str
            else:
                image_path = Path(image_path_str)

            if image_path.exists() and image_path.is_file():
                try:
                    base64_content = self._read_file_as_base64(image_path)
                    mime_type = self._get_image_mime_type(image_path)
                    data_url = f"data:{mime_type};base64,{base64_content}"

                    logger.debug(f"Replaced image: {image_path.name}")
                    return f"![{alt_text}]({data_url})"
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    return match.group(0)
            else:
                logger.warning(f"Image not found: {image_path}")
                return match.group(0)

        result = re.sub(image_pattern, replace_image, markdown_content)

        image_count = len(re.findall(image_pattern, markdown_content))
        if image_count > 0:
            logger.info(f"Processed {image_count} image references")

        return result

    def _split_content_into_pages(self, content: str) -> list[str]:
        pages = re.split("\n---\n", content)

        cleaned_pages = []
        for page in pages:
            cleaned_page = page.strip()
            if cleaned_page:
                cleaned_pages.append(cleaned_page.strip())

        if not cleaned_pages and content.strip():
            cleaned_pages = [content.strip()]

        logger.info(f"Split content into {len(cleaned_pages)} pages")
        return cleaned_pages

    def _page_to_doc(
        self,
        content: str,
        index: int,
        include_page_metadata: bool,
        extra_info: dict[str, Any] | None = None,
    ) -> Document:
        doc = Document(
            doc_id=str(uuid.uuid4()),
            text=content + "\n\n",
        )

        doc.metadata = extra_info or {}
        if include_page_metadata:
            doc.metadata[MetadataChunk.PAGE.value] = index + 1
            doc.excluded_llm_metadata_keys.append(MetadataChunk.PAGE.value)
            doc.excluded_embed_metadata_keys.append(MetadataChunk.PAGE.value)

        return doc

    def _get_slide_metadata(
        self, slide_index: int, exported_images: ExportedImages | None
    ) -> dict[str, Any]:
        """Extract slide-specific metadata from exported images."""
        metadata: dict[str, Any] = {}

        if not exported_images:
            return metadata

        if slide_index in exported_images.slide_images:
            slide_image_path = exported_images.slide_images[slide_index]
            metadata["doc_image"] = self._read_file_as_base64(slide_image_path)
            logger.debug(f"Added slide image metadata for slide {slide_index}")

        zones_for_slide = [
            zone
            for zone in exported_images.content_zones
            if zone.slide_index == slide_index
        ]

        if zones_for_slide:
            metadata["zones"] = zones_for_slide
            logger.debug(
                f"Added {len(zones_for_slide)} zones with full metadata for slide {slide_index}"
            )

        return metadata

    def _create_docs(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None,
        temp_dir: Path,
        exported_images: ExportedImages | None = None,
    ) -> list[Document]:
        output_md = temp_dir / "output.md"
        image_dir = temp_dir / "img"

        convert(
            pptx_path=str(file_info.file_data.absolute()),
            output=str(output_md),
            image_dir=str(image_dir),
            disable_image=False,
            disable_wmf=False,
            disable_notes=True,
            disable_escaping=True,
            disable_color=True,
            enable_slides=True,
            min_block_size=5,
        )

        with open(output_md, encoding="utf-8") as f:
            text = f.read()

        processed_text = self._replace_images_with_base64(text, temp_dir)
        pages = self._split_content_into_pages(processed_text)

        docs = []
        for index, page_content in enumerate(pages):
            slide_metadata = self._get_slide_metadata(index, exported_images)
            merged_extra_info = {**(extra_info or {}), **slide_metadata}

            doc = self._page_to_doc(
                content=page_content,
                index=index,
                include_page_metadata=True,
                extra_info=merged_extra_info,
            )
            docs.append(doc)

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

        # Calculate the mode of the transformation
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

        temp_dir = Path(
            os.fsdecode(
                await asyncio.to_thread(tempfile.mkdtemp, prefix="pptx_converter_")
            )
        )
        docs: list[Document] = []

        try:
            docs = await asyncio.to_thread(
                self._create_docs,
                file_info=file_info,
                extra_info=extra_info,
                temp_dir=temp_dir,
            )

            if self._reader_settings.vision.is_enabled:
                if notification:
                    notification(
                        percentage=0,
                        warnings=[IngestionParseErrors.USING_VLM_FOR_EXTRACTION],
                    )

                transform = (
                    PPTXSlideToImageDeepTransform()
                    if vision_mode == "deep"
                    else PPTXSlideToImageTransform()
                )

                (
                    converted_file_info,
                    exported_images,
                ) = await transform.transform_file(file_info)
                target_file_info = converted_file_info or file_info
                if exported_images:
                    converted_docs = self._create_docs(
                        file_info=target_file_info,
                        extra_info=extra_info,
                        temp_dir=temp_dir,
                        exported_images=exported_images,
                    )
                    if converted_docs:
                        docs = converted_docs
                elif (
                    converted_file_info
                    and converted_file_info.file_data != file_info.file_data
                ):
                    converted_docs = await asyncio.to_thread(
                        self._create_docs,
                        file_info=converted_file_info,
                        extra_info=extra_info,
                        temp_dir=temp_dir,
                    )
                    if converted_docs:
                        docs = converted_docs

            logger.info(f"Created {len(docs)} documents from {file_info.file_name}")

        finally:
            try:
                if temp_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")

        if not execute_transformations:
            logger.debug(
                "Skipping transformations for file: %s",
                file_info.file_name,
            )
            for node in docs:
                yield node
            return

        logger.debug(
            "Starting PPTX2md API transformations of file: %s",
            file_info.file_name,
        )

        for transformed_node in await arun_transformations(
            nodes=docs,
            transformations=list(
                slides_transformations(
                    reader_settings=self._reader_settings, vision_mode=vision_mode
                )
            ),
        ):
            yield transformed_node

        logger.debug(
            "Finished PPTX2md parsing and transformations of file: %s",
            file_info.file_name,
        )
