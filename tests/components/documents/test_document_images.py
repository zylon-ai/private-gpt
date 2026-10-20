from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

from llama_index.core.schema import BaseNode, MetadataMode

from private_gpt.components.chat.processors.chat_history.documents.document_images import (
    resolve_document_images,
)
from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER
from private_gpt.settings.settings import (
    TransformationReadersSettings,
    VisionSettings,
)

_IMAGE = "![Image](data:image/png;base64,aGVsbG8=)"
_TRANSFORM = (
    "private_gpt.components.chat.processors.chat_history.documents."
    "document_images.DescribeImageTransform"
)


def _vision() -> TransformationReadersSettings:
    return TransformationReadersSettings(vision=VisionSettings(mode="lite"))


class _FakeDescribe:
    """Replaces every embedded image with a fixed description."""

    def __init__(self, description: str | None) -> None:
        self._description = description

    async def acall(self, nodes: Sequence[BaseNode], **_: Any) -> Sequence[BaseNode]:
        if self._description is not None:
            for node in nodes:
                content = node.get_content(MetadataMode.NONE)
                node.set_content(content.replace(_IMAGE, self._description))
        return nodes


async def test_text_without_images_is_untouched() -> None:
    with patch(_TRANSFORM) as transform:
        assert await resolve_document_images("plain", _vision()) == "plain"
    transform.from_defaults.assert_not_called()


async def test_images_become_placeholders_when_description_is_off() -> None:
    with patch(_TRANSFORM) as transform:
        result = await resolve_document_images(f"before {_IMAGE} after")
    transform.from_defaults.assert_not_called()
    assert result == f"before {IMAGE_PLACEHOLDER} after"


async def test_images_are_described_when_description_is_on() -> None:
    with patch(_TRANSFORM) as transform:
        transform.from_defaults.return_value = _FakeDescribe("a red chart")
        result = await resolve_document_images(f"before {_IMAGE} after", _vision())
    assert result == "before a red chart after"
    assert "base64" not in result


async def test_undescribed_images_still_become_placeholders() -> None:
    # e.g. no model supports images, so the transform leaves the nodes as they are
    with patch(_TRANSFORM) as transform:
        transform.from_defaults.return_value = _FakeDescribe(None)
        result = await resolve_document_images(_IMAGE, _vision())
    assert result == IMAGE_PLACEHOLDER
