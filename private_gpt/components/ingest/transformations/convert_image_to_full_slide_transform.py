import base64
import io
import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from PIL import Image, ImageDraw

from private_gpt.settings.settings import (
    TransformationReadersSettings,
    settings,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


class ConvertImageToFullSlideTransform(TransformComponent):
    def __init__(
        self,
        reader_settings: TransformationReadersSettings,
        border_color: str = "red",
        border_width: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._regex = re.compile(r"!\[[^\]]*\]\(data:image/([^;]+);base64,([^)]+)\)")
        self.reader_settings = reader_settings
        self._border_color = border_color
        self._border_width = border_width
        self._kwargs = kwargs

    @classmethod
    def from_defaults(
        cls,
        reader_settings: TransformationReadersSettings,
        border_color: str = "red",
        border_width: int = 5,
    ) -> "ConvertImageToFullSlideTransform":
        return cls(
            reader_settings=reader_settings,
            border_color=border_color,
            border_width=border_width,
        )

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return list(self.transform(nodes, **kwargs))

    def transform(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        enhanced_nodes = []

        for node in nodes:
            enhanced_node = self._process_node(node)
            enhanced_nodes.append(enhanced_node)

        return enhanced_nodes

    def _process_node(self, node: BaseNode) -> BaseNode:
        metadata = node.metadata

        if "slide_image" not in metadata or "zones" not in metadata:
            logger.debug("Node missing slide_image or zones metadata, skipping")
            return node

        slide_image_b64 = metadata["slide_image"]
        zones = metadata["zones"]

        if not zones:
            logger.debug("No zones found in metadata, skipping")
            del metadata["slide_image"]
            del metadata["zones"]
            return node

        try:
            slide_image = self._decode_base64_image(slide_image_b64)
        except Exception as e:
            logger.error(f"Failed to decode slide image: {e}")
            del metadata["slide_image"]
            del metadata["zones"]
            return node

        content = node.get_content(MetadataMode.NONE)

        bordered_slides_cache: dict[int, str] = {}

        enhanced_content = self._replace_images_with_bordered_slides(
            content, slide_image, zones, bordered_slides_cache
        )

        node.set_content(enhanced_content)

        del metadata["slide_image"]
        del metadata["zones"]

        return node

    def _replace_images_with_bordered_slides(
        self,
        content: str,
        slide_image: Image.Image,
        zones: list[Any],
        cache: dict[int, str],
    ) -> str:
        images_in_content = list(re.finditer(self._regex, content))

        if len(images_in_content) != len(zones):
            logger.warning(
                f"Mismatch: {len(images_in_content)} images in content, "
                f"{len(zones)} zones in metadata"
            )

        zone_index = 0

        def replace_match(match: re.Match[str]) -> str:
            nonlocal zone_index

            if zone_index >= len(zones):
                return match.group(0)

            zone = zones[zone_index]
            zone_id = zone.zone_id

            if zone_id in cache:
                bordered_b64 = cache[zone_id]
            else:
                try:
                    bordered_slide = self._create_bordered_slide(slide_image, zone)
                    bordered_b64 = self._encode_image_to_base64(bordered_slide)
                    cache[zone_id] = bordered_b64
                    logger.debug(f"Created bordered slide for zone {zone_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to create bordered slide for zone {zone_id}: {e}"
                    )
                    zone_index += 1
                    return match.group(0)

            zone_index += 1
            return f"![](data:image/png;base64,{bordered_b64})"

        return re.sub(self._regex, replace_match, content)

    def _create_bordered_slide(
        self, slide_image: Image.Image, zone: Any
    ) -> Image.Image:
        bordered_image = slide_image.copy()
        draw = ImageDraw.Draw(bordered_image)

        slide_width, slide_height = bordered_image.size

        emu_to_px_factor = 144 / 914400

        left_px = int(zone.left * emu_to_px_factor)
        top_px = int(zone.top * emu_to_px_factor)
        right_px = int((zone.left + zone.width) * emu_to_px_factor)
        bottom_px = int((zone.top + zone.height) * emu_to_px_factor)

        left_px = max(0, min(left_px, slide_width))
        top_px = max(0, min(top_px, slide_height))
        right_px = max(0, min(right_px, slide_width))
        bottom_px = max(0, min(bottom_px, slide_height))

        for i in range(self._border_width):
            draw.rectangle(
                [left_px - i, top_px - i, right_px + i, bottom_px + i],
                outline=self._border_color,
                width=1,
            )

        logger.debug(
            f"Drew border at ({left_px}, {top_px}, {right_px}, {bottom_px}) "
            f"for zone {zone.zone_id}"
        )

        return bordered_image

    def _decode_base64_image(self, b64_string: str) -> Image.Image:
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
