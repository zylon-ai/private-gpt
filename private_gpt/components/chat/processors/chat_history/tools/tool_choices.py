from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ContentBlock,
    MessageRole,
)
from llama_index.core.base.llms.types import (
    TextBlock as LITextBlock,
)

from private_gpt.components.chat.models.chat_config_models import ToolSpec

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import (
        ContentBlock,
    )


def get_available_tools(
    tools: Sequence[ToolSpec],
    tool_choices: str | list[str],
) -> Sequence[ToolSpec]:
    if tool_choices in ("auto", "any"):
        return tools
    elif isinstance(tool_choices, str):
        return [tool for tool in tools if tool.name == tool_choices]
    elif isinstance(tool_choices, list):
        return [tool for tool in tools if tool.name in tool_choices]
    else:
        raise ValueError(
            f"Invalid tool_choices value: {tool_choices}. Must be 'auto', a string, or a list of strings."
        )


def _get_forced_suffix(
    tool_choices: str | list[str],
    tools: Sequence[ToolSpec],
) -> str | None:
    if tool_choices == "any":
        return "Always use one of the available tools to answer your question."
    if len(tools) == 1:
        return f"Always use the tool {tools[0].name} to answer your question."
    return None


def _add_suffix_to_last_user_message(
    chat_history: Sequence[ChatMessage],
    suffix: str,
) -> Sequence[ChatMessage]:
    last_user_message = next(
        (msg for msg in reversed(chat_history) if msg.role == MessageRole.USER),
        None,
    )
    last_message = chat_history[-1]

    if not last_user_message or last_user_message.content != last_message.content:
        return chat_history

    final_blocks: list[ContentBlock] = []
    added_suffix = False

    for block in reversed(last_message.blocks):
        if isinstance(block, LITextBlock) and not added_suffix:
            if block.text.endswith(suffix):
                return chat_history
            final_blocks.append(LITextBlock(text=f"{block.text}. {suffix}"))
            added_suffix = True
        else:
            final_blocks.append(block)

    last_message.blocks = list(reversed(final_blocks))
    return chat_history


async def process_tool_choices(
    chat_history: Sequence[ChatMessage] | None,
    tools: Sequence[ToolSpec] | None,
    tool_choices: str | list[str],
    **kwargs: Any,
) -> tuple[
    Sequence[ChatMessage] | None, Sequence[ToolSpec] | None, str | list[str] | None
]:
    if not chat_history or not tools or not tool_choices or tool_choices == "auto":
        return chat_history, tools, None

    if tool_choices == "none":
        return chat_history, [], None

    if tool_choices == "any" and chat_history[-1].role != MessageRole.USER:
        return chat_history, tools, "auto"

    filtered = get_available_tools(tools, tool_choices)
    suffix = _get_forced_suffix(tool_choices, filtered)
    if suffix:
        chat_history = _add_suffix_to_last_user_message(chat_history, suffix)

    return chat_history, filtered, None
