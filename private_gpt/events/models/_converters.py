from typing import Any, get_args

from llama_index.core.base.llms.types import AudioBlock as LIAudioBlock
from llama_index.core.base.llms.types import ContentBlock
from llama_index.core.base.llms.types import ImageBlock as LIImageBlock
from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.schema import NodeWithScore
from PIL.Image import Image

from private_gpt.components.chunk.models import Website
from private_gpt.events.models._base import BaseContentBlock
from private_gpt.events.models._content_blocks import (
    AudioBlock,
    ImageBlock,
    ResultContentBlockType,
    SourceBlock,
    TextBlock,
)
from private_gpt.server.mcp.mcp_service import (
    convert_mcp_blocks_to_llama_index,
    get_mcp_tool_result_content,
    is_mcp_content_block,
    is_mcp_tool_result,
)


def from_tool_output(tool_output: Any) -> list[ResultContentBlockType]:
    """Convert arbitrary tool output to a list of ``ResultContentBlockType`` blocks."""
    match tool_output:
        case list() if tool_output and all(
            isinstance(i, NodeWithScore) for i in tool_output
        ):
            return [SourceBlock.from_nodes(tool_output)]

        case list() if tool_output and all(isinstance(i, Website) for i in tool_output):
            return [SourceBlock.from_sources(tool_output)]

        case list():
            return [block for item in tool_output for block in from_tool_output(item)]

        case BaseContentBlock():
            if not any(
                isinstance(tool_output, t) for t in get_args(ResultContentBlockType)
            ):
                raise RuntimeError(
                    f"{type(tool_output).__name__} is not a member of ResultContentBlockType"
                )
            return [tool_output]

        case Image():
            return [ImageBlock.from_image(tool_output)]

        case LITextBlock():
            return [TextBlock(text=tool_output.text)]

        case LIImageBlock():
            image_b64 = tool_output.image_to_base64.decode()
            return [
                ImageBlock.from_base64(
                    data=image_b64,
                    mime_type=tool_output.image_mimetype or "image/png",
                )
            ]

        case LIAudioBlock():
            audio_b64 = tool_output.audio_to_base64.decode()
            return [
                AudioBlock.from_base64(
                    data=audio_b64,
                    mime_type=tool_output.format or "audio/mpeg",
                )
            ]

        case _ if is_mcp_content_block(tool_output):
            li_block = convert_mcp_blocks_to_llama_index(tool_output)
            return from_tool_output(li_block) if li_block else []

        case _ if is_mcp_tool_result(tool_output):
            return from_tool_output(get_mcp_tool_result_content(tool_output) or [])

        case _:
            return [TextBlock(text=str(tool_output))]


def to_llama_index_blocks(tool_output: Any) -> list[ContentBlock]:
    """Convert tool output to a list of LlamaIndex ``ContentBlock`` objects."""
    li_blocks: list[ContentBlock] = []
    for block in from_tool_output(tool_output):
        if isinstance(block, TextBlock | ImageBlock | AudioBlock):
            li_blocks.append(block.to_llama_index())
    return li_blocks
