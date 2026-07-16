import base64
import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if not settings().server.debug_mode else logging.DEBUG)


class ImageComparer:
    """Helper class for comparing images to detect duplicates."""

    @staticmethod
    def get_image_difference(image_bytes_1: bytes, image_bytes_2: bytes) -> float:
        """Compare two images and return their difference score."""
        try:
            from io import BytesIO

            import imagehash  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            from PIL import Image

            hash0 = imagehash.average_hash(Image.open(BytesIO(image_bytes_1)))
            hash1 = imagehash.average_hash(Image.open(BytesIO(image_bytes_2)))

            diff: float = hash0 - hash1
            return diff

        except Exception as e:
            logger.warning(f"Error comparing images: {e}")
            return 10000  # Return high difference on error


class ImageDeduplicationTransform(TransformComponent):
    """Transform component that deduplicates similar images across all nodes."""

    def __init__(
        self,
        similarity_threshold: float | None = None,
        preserve_alt_text: bool = True,
    ):
        super().__init__()
        self._similarity_threshold = similarity_threshold or 1.0
        self._preserve_alt_text = preserve_alt_text
        self._regex = re.compile(r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)")

    @classmethod
    def from_defaults(
        cls,
        similarity_threshold: float | None = None,
        preserve_alt_text: bool = True,
    ) -> "ImageDeduplicationTransform":
        return cls(similarity_threshold, preserve_alt_text)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return list(self._deduplicate_images_across_nodes(nodes))

    def _deduplicate_images_across_nodes(
        self, nodes: Sequence[BaseNode]
    ) -> Sequence[BaseNode]:
        if not nodes:
            return nodes

        logger.info(f"Starting image deduplication across {len(nodes)} nodes")

        all_images = self._extract_all_images(nodes)
        if not all_images:
            logger.info("No images found for deduplication")
            return nodes

        logger.info(f"Found {len(all_images)} total images across all nodes")

        duplicate_groups = self._find_duplicate_groups(all_images)
        replacement_map = self._create_replacement_mapping(duplicate_groups, all_images)
        processed_nodes = [
            self._apply_replacements_to_node(node, replacement_map) for node in nodes
        ]

        total_images = len(all_images)
        unique_images = len(duplicate_groups)
        duplicates_removed = total_images - unique_images

        logger.info(
            f"Deduplication complete: {total_images} total images, "
            f"{unique_images} unique, {duplicates_removed} duplicates removed "
            f"({duplicates_removed / total_images * 100:.1f}% reduction)"
        )

        return processed_nodes

    def _extract_all_images(
        self, nodes: Sequence[BaseNode]
    ) -> list[tuple[int, re.Match[str], bytes, str, str]]:
        all_images = []

        for node_idx, node in enumerate(nodes):
            content = node.get_content(MetadataMode.NONE)
            matches = list(re.finditer(self._regex, content))

            for match in matches:
                alt_text, mime_type, b64_content = match.groups()

                if self._is_valid_base64(b64_content):
                    try:
                        image_bytes = base64.b64decode(b64_content)
                        all_images.append(
                            (node_idx, match, image_bytes, alt_text, mime_type)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image: {e}")
                        continue

        return all_images

    def _find_duplicate_groups(
        self, all_images: list[tuple[int, re.Match[str], bytes, str, str]]
    ) -> list[list[int]]:
        duplicate_groups = []
        processed = set()

        for i, (_, _, image_bytes_i, _, _) in enumerate(all_images):
            if i in processed:
                continue

            # Start a new group with this image
            current_group = [i]
            processed.add(i)

            # Find all similar images
            for j, (_, _, image_bytes_j, _, _) in enumerate(all_images[i + 1 :], i + 1):
                if j in processed:
                    continue

                try:
                    diff = ImageComparer.get_image_difference(
                        image_bytes_i, image_bytes_j
                    )
                    if diff < self._similarity_threshold:
                        current_group.append(j)
                        processed.add(j)
                        logger.debug(
                            f"Found similar images: index {i} ~ index {j} (diff: {diff:.3f})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error comparing images at indices {i} and {j}: {e}"
                    )
                    continue

            duplicate_groups.append(current_group)

        return duplicate_groups

    def _create_replacement_mapping(
        self,
        duplicate_groups: list[list[int]],
        all_images: list[tuple[int, re.Match[str], bytes, str, str]],
    ) -> dict[str, str]:
        replacement_map = {}

        for group in duplicate_groups:
            if len(group) <= 1:
                continue

            # Use the first image in the group as canonical
            canonical_idx = group[0]
            canonical_image = all_images[canonical_idx]
            (
                _,
                _,
                canonical_bytes,
                canonical_alt,
                canonical_mime,
            ) = canonical_image

            # Create canonical data URL
            canonical_b64 = base64.b64encode(canonical_bytes).decode()
            canonical_data_url = f"data:{canonical_mime};base64,{canonical_b64}"

            # Map all images in the group to the canonical version
            for img_idx in group:
                _, match, _, alt_text, _ = all_images[img_idx]
                original_b64 = match.groups()[2]

                # Preserve original alt text if requested
                final_alt_text = alt_text if self._preserve_alt_text else canonical_alt
                final_replacement = f"![{final_alt_text}]({canonical_data_url})"

                replacement_map[original_b64] = final_replacement

        return replacement_map

    def _apply_replacements_to_node(
        self, node: BaseNode, replacement_map: dict[str, str]
    ) -> BaseNode:
        """Apply image replacements to a single node."""
        content = node.get_content(MetadataMode.NONE)

        def replace_image(match: re.Match[str]) -> str:
            b64_content = match.groups()[2]
            if b64_content in replacement_map:
                return replacement_map[b64_content]
            return match.group(0)  # Return original if no replacement

        enhanced_content = re.sub(self._regex, replace_image, content)
        node.set_content(enhanced_content)

        return node

    @staticmethod
    def _is_valid_base64(s: str) -> bool:
        """Check if string is valid base64."""
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
