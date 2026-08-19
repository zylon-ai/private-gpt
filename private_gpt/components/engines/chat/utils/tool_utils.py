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


def _stream_tool_call_id(tool_call: Any) -> str | None:
    """Return a stable id from ToolSelection (``tool_id``) or OpenAI deltas (``id``)."""
    if isinstance(tool_call, dict):
        ident = tool_call.get("tool_id") or tool_call.get("id")
    else:
        # Pydantic OpenAI models raise AttributeError for unknown fields, so
        # getattr with a default is required (``tc.tool_id`` crashes).
        ident = getattr(tool_call, "tool_id", None) or getattr(tool_call, "id", None)
    if ident is None:
        return None
    ident_str = str(ident).strip()
    return ident_str or None


def _stream_tool_call_index(tool_call: Any) -> int | None:
    if isinstance(tool_call, dict):
        index = tool_call.get("index")
    else:
        index = getattr(tool_call, "index", None)
    return index if isinstance(index, int) else None


def _accumulate_openai_tool_call(current: Any, delta: Any) -> Any:
    """Fold an OpenAI ``ChoiceDeltaToolCall`` argument/name fragment into *current*."""
    current_fn = getattr(current, "function", None)
    delta_fn = getattr(delta, "function", None)
    if current_fn is None or delta_fn is None:
        return current

    if current_fn.arguments is None:
        current_fn.arguments = ""
    if current_fn.name is None:
        current_fn.name = ""

    current_fn.arguments += delta_fn.arguments or ""
    current_fn.name += delta_fn.name or ""
    delta_id = getattr(delta, "id", None) or ""
    if delta_id:
        current.id = (getattr(current, "id", None) or "") + delta_id
    delta_type = getattr(delta, "type", None)
    if delta_type and getattr(current, "type", None) is None:
        current.type = delta_type
    return current


def merge_stream_tool_calls(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Merge streamed tool-call payloads from consecutive LLM chunks.

    Providers typically emit either:

    - ``ToolSelection`` objects keyed by ``tool_id``
    - OpenAI ``ChoiceDeltaToolCall`` objects keyed by ``id`` (and ``index``)

    LlamaIndex's OpenAI adapter already accumulates argument fragments before
    yielding, so later snapshots with the same id replace earlier ones. Raw
    OpenAI deltas without an id (argument-only fragments) are folded onto the
    matching ``index`` so the complete call can be sent back on the next turn.
    """
    merged: list[Any] = list(existing)

    def _find(ident: str | None, index: int | None) -> int | None:
        for i, tool_call in enumerate(merged):
            if ident and _stream_tool_call_id(tool_call) == ident:
                return i
            if (
                ident is None
                and index is not None
                and _stream_tool_call_index(tool_call) == index
            ):
                return i
        return None

    for tool_call in incoming:
        ident = _stream_tool_call_id(tool_call)
        index = _stream_tool_call_index(tool_call)
        pos = _find(ident, index)
        if pos is None:
            merged.append(tool_call)
            continue
        if ident:
            # Identified snapshot (ToolSelection or accumulated OpenAI call).
            merged[pos] = tool_call
        else:
            merged[pos] = _accumulate_openai_tool_call(merged[pos], tool_call)
    return merged


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
