# SPDX-License-Identifier: Apache-2.0

import json
from json import JSONDecodeError, JSONDecoder
from typing import Any

import partial_json_parser  # type: ignore
from partial_json_parser.core.exceptions import MalformedJSON  # type: ignore
from partial_json_parser.core.options import Allow  # type: ignore

_KNOWN_EXTRA_KEYS = frozenset({"tool_calls", "tool_call_id"})


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _format_hf_content_block(block: dict[str, Any]) -> str:
    """Render a single HF-format content dict as a plain string.

    Handles all block types that ChatTemplatePromptStyle may produce when
    content_format="openai", plus common OpenAI-compatible types.
    """
    btype = block.get("type", "text")

    if btype == "text":
        text: str = block.get("text", "")
        return text

    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url", "?")
        return f"[image: url={url}]"

    if btype == "image":
        source = block.get("source") or {}
        if source.get("type") == "url":
            return f"[image: url={source.get('url', '?')}]"
        mime = source.get("media_type", "image/*")
        return f"[image: {mime} (base64)]"

    if btype == "audio":
        source = block.get("source") or {}
        if source.get("type") == "url":
            return f"[audio: url={source.get('url', '?')}]"
        mime = source.get("media_type", "audio/*")
        return f"[audio: {mime} (base64)]"

    if btype == "thinking":
        content = (block.get("thinking") or "").strip()
        return f"<thinking>\n{content}\n</thinking>" if content else "<thinking/>"

    if btype == "redacted_thinking":
        return "<thinking>[redacted]</thinking>"

    if btype in ("tool_use", "server_tool_use"):
        args = _safe_json(block.get("input", {}))
        return f"[tool_use: id={block.get('id')} name={block.get('name')} input={args}]"

    if btype == "tool_call":
        args = _safe_json(block.get("tool_kwargs", {}))
        return f"[tool_use: id={block.get('tool_call_id')} name={block.get('tool_name')} input={args}]"

    if btype == "tool_result":
        inner = block.get("content", "")
        if isinstance(inner, list):
            inner = "\n".join(
                b.get("text", "") for b in inner if b.get("type") == "text"
            )
        return f"[tool_result: tool_use_id={block.get('tool_use_id')}]\n{inner}\n[/tool_result]"

    return f"[{btype}]"


def format_hf_conversation(
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render an HF-format conversation as a ChatML string.

    Mirrors what an HF tokenizer chat template would produce. Designed to
    handle the dict format emitted by ChatTemplatePromptStyle._to_hf_messages,
    where content is a plain string and tool_calls / tool_call_id may appear
    as extra keys spread from message.additional_kwargs.
    """
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            body = "\n".join(s for b in content if (s := _format_hf_content_block(b)))
        else:
            body = str(content)

        # tool_calls from additional_kwargs (assistant messages using tools)
        if tool_calls := msg.get("tool_calls"):
            body = f"{body}\n[tool_calls: {_safe_json(tool_calls)}]".lstrip("\n")

        # tool_call_id from additional_kwargs (tool result messages)
        if tool_call_id := msg.get("tool_call_id"):
            body = f"[tool_call_id={tool_call_id}]\n{body}"

        parts.append(f"<|im_start|>{role}\n{body}\n<|im_end|>")

    if tools:
        parts.append(f"[tools: {_safe_json(tools)}]")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


# partial_json_parser doesn't support extra data and
# JSONDecorder.raw_decode doesn't support partial JSON
def partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (MalformedJSON, JSONDecodeError, json.JSONDecodeError) as e:
        message = getattr(e, "msg", None)
        if isinstance(message, str) and "Extra data" in message:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise
    except Exception as e:
        raise MalformedJSON(f"Failed to parse JSON: {e!s}") from e
