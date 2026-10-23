import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER

logger = logging.getLogger(__name__)


class ReplaceImageByPlaceholder(TransformComponent):
    def __init__(self) -> None:
        super().__init__()
        self._regex = re.compile(r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)")

    @classmethod
    def from_defaults(cls) -> "ReplaceImageByPlaceholder":
        return cls()

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return self._replace_images_by_placeholder(nodes)

    def _replace_images_by_placeholder(
        self, nodes: Sequence[BaseNode]
    ) -> Sequence[BaseNode]:
        for node in nodes:
            content = node.get_content(MetadataMode.NONE)
            matches = list(re.finditer(self._regex, content))

            for match in matches:
                content = content.replace(match.group(0), IMAGE_PLACEHOLDER)
                node.set_content(content)

        return nodes
