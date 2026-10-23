from collections.abc import Callable
from typing import Any, cast, get_args

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

NO_TOOL_CONTENT = "(no-output)"


def _rendered_text(block: ResultContentBlockType) -> str:
    if isinstance(block, TextBlock):
        return block.text
    render = getattr(block, "render", None)
    if callable(render):
        render_fn = cast(Callable[[], str], render)
        try:
            return str(render_fn())
        except Exception:
            return ""
    return ""


def normalize_tool_result_content(
    content: list[ResultContentBlockType] | None,
) -> list[ResultContentBlockType]:
    """Ensure an empty tool result has model-visible text content.

    Builders should write ``NO_TOOL_CONTENT`` into their native result
    payload.  This helper is only a fallback: fill empty text blocks, and
    if the result is still empty, append a sibling text block so prompt
    formatting cannot drop the TOOL message.
    """
    blocks: list[ResultContentBlockType] = []
    for block in content or []:
        if isinstance(block, TextBlock) and not block.text.strip():
            blocks.append(block.model_copy(update={"text": NO_TOOL_CONTENT}))
        else:
            blocks.append(block)
    if not blocks:
        return [TextBlock(text=NO_TOOL_CONTENT)]
    if not any(_rendered_text(block).strip() for block in blocks):
        blocks.append(TextBlock(text=NO_TOOL_CONTENT))
    return blocks


def from_tool_output(tool_output: Any) -> list[ResultContentBlockType]:
    """Convert arbitrary tool output to a list of ``ResultContentBlockType`` blocks."""
    match tool_output:
        case None:
            return []

        case str() if not tool_output.strip():
            return []

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
            continue
        rendered = _rendered_text(block)
        if rendered.strip():
            li_blocks.append(LITextBlock(text=rendered))
    return li_blocks
