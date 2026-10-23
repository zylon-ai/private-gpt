import json
from collections.abc import Sequence
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from private_gpt.components.engines.citations.format import format_context
from private_gpt.components.engines.citations.types import Document
from private_gpt.events.models import SourceBlock


def _get_custom_content_by_type(t: str, content: Any) -> str | None:
    match t:
        case "source":
            source_blocks = cast(list[SourceBlock], content)
            try:
                documents = [
                    Document.from_source(source)
                    for block in source_blocks
                    for source in block.sources
                ]
                documents, result = format_context(documents, generate_citations=False)
                return result
            except AttributeError:
                # If the content is not a list of SourceBlock, return None
                return None
        case "tool_calls":
            if not isinstance(content, list):
                return None
            elements: list[dict[str, Any]] = []
            for element in content:
                if isinstance(element, BaseModel):
                    elements.append(element.model_dump())
                elif isinstance(element, dict):
                    elements.append(element)
            return json.dumps(elements)
        case _:
            return None


def _get_custom_content(
    message: ChatMessage,
) -> str | None:
    """Get custom content from a ChatMessage."""
    if not message.additional_kwargs:
        return None

    for key, value in message.additional_kwargs.items():
        custom_content = _get_custom_content_by_type(key, value)
        if custom_content is not None:
            return custom_content

    return None


def messages_to_history_str(
    messages: Sequence[ChatMessage], show_index: bool = False, show_role: bool = True
) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for i, message in enumerate(messages):
        custom_content = _get_custom_content(message)
        content = custom_content or message.content or ""

        prefix = f"[{i}] " if show_index else ""
        role = f"{message.role.value}: " if show_role else ""

        string_message = f"{prefix}{role}{content}"
        string_messages.append(string_message)

    return "\n".join(string_messages)
