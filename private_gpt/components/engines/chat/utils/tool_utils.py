import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.agent.workflow.workflow_events import ToolCallResult
from llama_index.core.base.llms.types import (
    AudioBlock,
    ImageBlock,
    TextBlock,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import AsyncBaseTool, ToolOutput

from private_gpt.events.models import (
    ContentBlockType,
    from_tool_output,
    to_llama_index_blocks,
)
from private_gpt.server.mcp.mcp_service import (
    convert_mcp_blocks_to_llama_index,
    get_mcp_tool_result_content,
    is_mcp_tool_result,
)

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import (
        ContentBlock,
    )

logger = logging.getLogger(__name__)


def select_tool_names(
    tool_choices: str | list[str], tool_names: list[str]
) -> list[str]:
    """Filter tool names according to tool choice policy."""
    if tool_choices in ("auto", "any"):
        return tool_names
    if isinstance(tool_choices, str):
        return [name for name in tool_names if name == tool_choices]
    return [name for name in tool_names if name in tool_choices]


async def execute_tool_call(
    tool: AsyncBaseTool,
    tool_name: str,
    tool_id: str,
    tool_kwargs: dict[str, Any],
    state_ctx: Any,
) -> tuple[ToolCallResult, ChatMessage]:
    """Execute one tool call and convert output into tool message blocks."""
    try:
        if getattr(tool, "requires_context", False):
            context_tool: Any = tool
            tool_output = await context_tool.acall(ctx=state_ctx, **tool_kwargs)
        else:
            tool_output = await tool.acall(**tool_kwargs)
    except Exception as error:
        logger.exception("Tool execution failed for %s", tool_name)
        tool_output = ToolOutput(
            content=str(error),
            tool_name=tool_name,
            raw_input=tool_kwargs,
            raw_output=str(error),
            is_error=True,
        )

    # Double check that content is stored in blocks, not as content string
    # Llama Index always converts blocks to string content...
    if tool_output.raw_output and isinstance(tool_output.raw_output, list):
        # We are returning directly a list of blocks
        content_blocks = tool_output.raw_output
        li_blocks: list[ContentBlock] = []

        for block in content_blocks:
            if isinstance(block, TextBlock | ImageBlock | AudioBlock):
                li_blocks.append(block)
            elif isinstance(block, ContentBlockType):
                li_blocks.extend(to_llama_index_blocks([block]))
            else:
                li_blocks.append(TextBlock(text=str(block)))

        tool_output.blocks = li_blocks
    elif is_mcp_tool_result(tool_output.raw_output):
        # Convert MCP result to LLama index
        converted_blocks: list[ContentBlock] = []
        for block in get_mcp_tool_result_content(tool_output.raw_output) or []:
            converted_block = convert_mcp_blocks_to_llama_index(block)
            if not converted_block:
                converted_block = TextBlock(text=str(block))
            converted_blocks.append(converted_block)
        tool_output.blocks = converted_blocks

    # Build the tool message
    tool_result_block = from_tool_output(tool_output.raw_output)
    unique_types = {result.type for result in tool_result_block}
    tool_result_block_map = {
        block_type: [block for block in tool_result_block if block.type == block_type]
        for block_type in unique_types
        # We already have the content in the blocks
        if block_type not in ("text", "image", "audio")
    }

    tool_message = ChatMessage(
        role="tool",
        content=tool_output.content,
        additional_kwargs={
            **tool_result_block_map,
            "tool_call_id": tool_id,
            "tool_call_name": tool_name,
            "tool_call_args": tool_kwargs,
            "raw_output": tool_output.raw_output,
        },
    )

    result = ToolCallResult(
        tool_name=tool_name,
        tool_kwargs=tool_kwargs,
        tool_id=tool_id,
        tool_output=tool_output,
        return_direct=tool.metadata.return_direct,
    )
    return result, tool_message
