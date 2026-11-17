"""Deterministic repeat tracking and the LLM-facing view used to break loops.

Small models continue the pattern they see in the chat history: once a few
identical tool calls (or identical tool results) pile up, they keep copying the
last call instead of using the result. Measured on Qwen 3.8 27B without
reasoning, rewriting the history the model sees breaks that pattern reliably:

* ``NOTE``     - annotate repeated tool results so the model notices them.
* ``FLATTEN``  - render the actions since the latest user message as a work log
                 inside one user message, so there is no call/result pattern to
                 continue. Tools stay available.
* ``FINAL``    - same work log, tools removed, and the model is asked to answer.

Only the view sent to the LLM changes; the stored history is never modified.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from pydantic import BaseModel, Field

from private_gpt.events.models import NO_TOOL_CONTENT

if TYPE_CHECKING:
    from collections.abc import Sequence

# Results shorter than this (``ok``, empty output, ...) are too generic to be
# treated as a repetition signal on their own.
_MIN_RESULT_SIGNAL_CHARS = 16
_MAX_ARG_CHARS = 400
_MAX_RESULT_CHARS = 1500
_WHITESPACE = re.compile(r"\s+")

REPEAT_NOTE = (
    "[Note: this repeats an earlier action or result from this request. "
    "Repeating it will not change anything; use the results you already have "
    "to answer the user, or change approach.]"
)

FLATTEN_INSTRUCTIONS = (
    "Below is the log of what you have already done for this request: your "
    "previous actions and their results. Review it, then decide the single best "
    "next step.\n"
    "- If the log already contains what you need, answer the user now without "
    "calling tools.\n"
    "- Never repeat an action that already failed or that already returned the "
    "same result; change approach."
)

FINAL_INSTRUCTIONS = (
    "You cannot call tools anymore. Using only the work log above, give the user "
    "the final answer now. If the task could not be completed, explain briefly "
    "what blocked it."
)


class LoopGuardLevel(enum.IntEnum):
    NONE = 0
    NOTE = 1
    FLATTEN = 2
    FINAL = 3


class LoopGuardDirective(BaseModel):
    """Per-iteration instruction for how the engine renders the LLM input.

    Recomputed before every iteration; never persisted into the history.
    """

    level: LoopGuardLevel = Field(default=LoopGuardLevel.NONE)
    reason: str = Field(default="")


@dataclass(frozen=True)
class ToolStep:
    """One tool call and its result after the latest user message."""

    tool_name: str
    call: str
    result: str
    call_signature: str
    result_signature: str | None


@dataclass
class RepeatAnalysis:
    steps: list[ToolStep] = field(default_factory=list)
    # Index (into ``steps``) of every step that repeats an earlier one.
    repeated_indexes: set[int] = field(default_factory=set)
    # Occurrences of the latest step's call or result (whichever is higher).
    latest_repeat_count: int = 0


def last_user_index(messages: Sequence[ChatMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == MessageRole.USER:
            return index
    return None


def _normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_args(args: Any) -> Any:
    """Decode JSON arguments and drop unset (``None``) parameters.

    Engines fill optional parameters with ``None`` after schema coercion; they
    carry no information and would make identical calls look different.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    if isinstance(args, dict):
        return {key: value for key, value in args.items() if value is not None}
    return args


def _canonical_args(args: Any) -> str:
    args = _parse_args(args)
    if isinstance(args, str):
        return _normalize_text(args)
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def _tool_call_args_by_id(messages: Sequence[ChatMessage]) -> dict[str, Any]:
    """Map tool call id to arguments from assistant ``tool_calls``."""
    by_id: dict[str, Any] = {}
    for message in messages:
        if message.role != MessageRole.ASSISTANT:
            continue
        for call in message.additional_kwargs.get("tool_calls") or []:
            if isinstance(call, dict):
                call_id = call.get("tool_id") or call.get("id")
                args = call.get("tool_kwargs")
                if args is None and isinstance(call.get("function"), dict):
                    args = call["function"].get("arguments")
            else:
                call_id = getattr(call, "tool_id", None) or getattr(call, "id", None)
                args = getattr(call, "tool_kwargs", None)
            if call_id:
                by_id[str(call_id)] = args
    return by_id


def _format_call(tool_name: str, args: Any) -> str:
    args = _parse_args(args)
    if isinstance(args, str):
        return f"{tool_name}({_truncate(args, _MAX_ARG_CHARS)})"
    if not isinstance(args, dict):
        return f"{tool_name}()" if args is None else f"{tool_name}({args!s})"
    rendered = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        rendered.append(
            f"{key}={json.dumps(_truncate(text, _MAX_ARG_CHARS), ensure_ascii=False)}"
        )
    return f"{tool_name}({', '.join(rendered)})"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def extract_tool_steps(messages: Sequence[ChatMessage]) -> list[ToolStep]:
    """Collect tool steps after the latest user message, in order."""
    start = last_user_index(messages)
    if start is None:
        return []
    window = messages[start + 1 :]
    args_by_id = _tool_call_args_by_id(window)

    steps: list[ToolStep] = []
    for message in window:
        if message.role != MessageRole.TOOL:
            continue
        kwargs = message.additional_kwargs
        tool_name = str(kwargs.get("tool_call_name") or kwargs.get("name") or "tool")
        args = kwargs.get("tool_call_args")
        if args is None:
            args = args_by_id.get(str(kwargs.get("tool_call_id") or ""))
        result_text = _normalize_text(message.content or "")
        has_signal = (
            len(result_text) >= _MIN_RESULT_SIGNAL_CHARS
            and result_text != NO_TOOL_CONTENT
        )
        steps.append(
            ToolStep(
                tool_name=tool_name,
                call=_format_call(tool_name, args),
                result=message.content or "",
                call_signature=_digest(f"{tool_name}\0{_canonical_args(args)}"),
                result_signature=(
                    _digest(f"{tool_name}\0{result_text}") if has_signal else None
                ),
            )
        )
    return steps


def analyze_repeats(messages: Sequence[ChatMessage]) -> RepeatAnalysis:
    """Count repeated calls and results since the latest user message.

    Counts occurrences, not consecutive runs: alternating between two
    identical calls is still a loop.
    """
    steps = extract_tool_steps(messages)
    analysis = RepeatAnalysis(steps=steps)
    call_counts: dict[str, int] = {}
    result_counts: dict[str, int] = {}
    for index, step in enumerate(steps):
        seen_call = call_counts.get(step.call_signature, 0)
        seen_result = (
            result_counts.get(step.result_signature, 0) if step.result_signature else 0
        )
        if seen_call or seen_result:
            analysis.repeated_indexes.add(index)
        call_counts[step.call_signature] = seen_call + 1
        if step.result_signature:
            result_counts[step.result_signature] = seen_result + 1

    if steps:
        latest = steps[-1]
        analysis.latest_repeat_count = max(
            call_counts[latest.call_signature],
            result_counts.get(latest.result_signature, 0)
            if latest.result_signature
            else 0,
        )
    return analysis


def _with_note(message: ChatMessage) -> ChatMessage:
    updated = message.model_copy(deep=False)
    updated.additional_kwargs = dict(message.additional_kwargs)
    updated.blocks = [TextBlock(text=f"{message.content or ''}\n\n{REPEAT_NOTE}")]
    return updated


def _annotate_repeats(
    messages: list[ChatMessage], analysis: RepeatAnalysis
) -> list[ChatMessage]:
    start = last_user_index(messages)
    if start is None or not analysis.repeated_indexes:
        return messages
    annotated = list(messages)
    step_index = -1
    for position in range(start + 1, len(annotated)):
        if annotated[position].role != MessageRole.TOOL:
            continue
        step_index += 1
        if step_index in analysis.repeated_indexes:
            annotated[position] = _with_note(annotated[position])
    return annotated


def render_work_log(analysis: RepeatAnalysis) -> str:
    lines = []
    for index, step in enumerate(analysis.steps):
        result = _truncate(step.result.strip() or NO_TOOL_CONTENT, _MAX_RESULT_CHARS)
        note = f"\n    {REPEAT_NOTE}" if index in analysis.repeated_indexes else ""
        lines.append(f"[{index + 1}] action: {step.call}\n    result: {result}{note}")
    return "\n".join(lines)


def _flatten(
    messages: list[ChatMessage],
    analysis: RepeatAnalysis,
    final: bool,
) -> list[ChatMessage]:
    start = last_user_index(messages)
    if start is None:
        return messages
    user = messages[start]
    text = (
        f"\n\n---\n{FLATTEN_INSTRUCTIONS}\n\n"
        f"<work_log>\n{render_work_log(analysis)}\n</work_log>"
    )
    if final:
        text += f"\n\n{FINAL_INSTRUCTIONS}"
    flattened = ChatMessage(
        role=MessageRole.USER,
        blocks=[*user.blocks, TextBlock(text=text)],
        additional_kwargs=dict(user.additional_kwargs),
    )
    return [*messages[:start], flattened]


def apply_loop_guard_view(
    messages: list[ChatMessage],
    directive: LoopGuardDirective | None,
) -> tuple[list[ChatMessage], bool]:
    """Return the messages to send to the LLM and whether tools are disabled."""
    if directive is None or directive.level == LoopGuardLevel.NONE:
        return messages, False
    analysis = analyze_repeats(messages)
    if not analysis.steps:
        return messages, False
    if directive.level == LoopGuardLevel.NOTE:
        return _annotate_repeats(messages, analysis), False
    final = directive.level >= LoopGuardLevel.FINAL
    return _flatten(messages, analysis, final=final), final
