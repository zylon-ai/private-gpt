"""Loop guard: repeat counting and the LLM-facing history view.

Fixtures mirror two loops captured from Qwen 3.8 27B:

* failed calls: the same ``create`` call without ``file_text`` returning the
  same error again and again;
* successful calls: the same script returning the same (correct) output while
  the model never answers.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.utils.loop_guard import (
    FINAL_INSTRUCTIONS,
    REPEAT_NOTE,
    LoopGuardDirective,
    LoopGuardLevel,
    analyze_repeats,
    apply_loop_guard_view,
)
from private_gpt.server.chat.interceptors.repeat_guard_interceptor import (
    RepeatGuardRequestInterceptor,
)
from private_gpt.settings.settings import RepeatGuardSettings

_CREATE_ARGS = {"command": "create", "path": "/workspace/lieferungen.csv"}
_CREATE_ERROR = "create requires the file_text parameter."
_SCRIPT_ARGS = {"command": "python3 calc.py"}
_SCRIPT_OUTPUT = "Total deliveries: 180\nSLA missed: 41\nClaimable: 1273.80"

Step = tuple[str, dict[str, Any], str]


def _call(call_id: str, name: str, args: dict[str, Any]) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        additional_kwargs={
            "tool_calls": [{"tool_id": call_id, "tool_name": name, "tool_kwargs": args}]
        },
    )


def _result(call_id: str, name: str, args: dict[str, Any], output: str) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content=output,
        additional_kwargs={
            "tool_call_id": call_id,
            "tool_call_name": name,
            "tool_call_args": args,
        },
    )


def _history(steps: list[Step], *, previous_turn: bool = False) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if previous_turn:
        messages += [
            ChatMessage(role=MessageRole.USER, content="earlier request"),
            _call("old", "bash_code_execution", _SCRIPT_ARGS),
            _result("old", "bash_code_execution", _SCRIPT_ARGS, _SCRIPT_OUTPUT),
            ChatMessage(role=MessageRole.ASSISTANT, content="earlier answer"),
        ]
    messages.append(ChatMessage(role=MessageRole.USER, content="Compute the penalty"))
    for index, (name, args, output) in enumerate(steps):
        call_id = f"t{index}"
        messages += [_call(call_id, name, args), _result(call_id, name, args, output)]
    return messages


def _failed_creates(count: int) -> list[Step]:
    return [("text_editor_code_execution", _CREATE_ARGS, _CREATE_ERROR)] * count


def _same_script(count: int) -> list[Step]:
    return [("bash_code_execution", _SCRIPT_ARGS, _SCRIPT_OUTPUT)] * count


# ---------------------------------------------------------------- analysis


def test_counts_only_steps_after_latest_user_message() -> None:
    analysis = analyze_repeats(_history(_same_script(1), previous_turn=True))

    assert len(analysis.steps) == 1
    assert analysis.repeated_indexes == set()
    assert analysis.latest_repeat_count == 1


def test_identical_failed_calls_are_repeats() -> None:
    analysis = analyze_repeats(_history(_failed_creates(4)))

    assert analysis.repeated_indexes == {1, 2, 3}
    assert analysis.latest_repeat_count == 4


def test_same_result_from_different_code_is_a_repeat() -> None:
    steps = [
        ("bash_code_execution", {"command": f"python3 v{n}.py"}, _SCRIPT_OUTPUT)
        for n in range(3)
    ]
    analysis = analyze_repeats(_history(steps))

    assert analysis.repeated_indexes == {1, 2}
    assert analysis.latest_repeat_count == 3


def test_argument_order_does_not_hide_repeats() -> None:
    steps: list[Step] = [
        ("tool", {"a": 1, "b": "x"}, "first output long enough"),
        ("tool", {"b": "x", "a": 1}, "second output long enough"),
    ]
    assert analyze_repeats(_history(steps)).repeated_indexes == {1}


def test_unset_parameters_are_ignored() -> None:
    padded = {**_CREATE_ARGS, "file_text": None, "view_range": None}
    steps: list[Step] = [
        ("text_editor_code_execution", _CREATE_ARGS, _CREATE_ERROR),
        ("text_editor_code_execution", padded, "a different error message here"),
    ]
    analysis = analyze_repeats(_history(steps))

    assert analysis.repeated_indexes == {1}
    assert "null" not in analysis.steps[1].call


def test_short_generic_results_are_not_a_repeat_signal() -> None:
    steps: list[Step] = [
        ("bash_code_execution", {"command": "mkdir a"}, "ok"),
        ("bash_code_execution", {"command": "mkdir b"}, "ok"),
    ]
    assert analyze_repeats(_history(steps)).repeated_indexes == set()


def test_progressing_work_is_not_a_repeat() -> None:
    steps: list[Step] = [
        ("bash_code_execution", {"command": "ls"}, "data.csv notes.md report.txt"),
        ("bash_code_execution", {"command": "head data.csv"}, "id,name\n1,alice"),
        ("bash_code_execution", {"command": "wc -l data.csv"}, "1532 data.csv lines"),
    ]
    assert analyze_repeats(_history(steps)).repeated_indexes == set()


def test_reads_args_from_assistant_tool_calls_when_result_has_none() -> None:
    messages = _history(_failed_creates(2))
    for message in messages:
        message.additional_kwargs.pop("tool_call_args", None)

    analysis = analyze_repeats(messages)

    assert analysis.steps[0].call.startswith("text_editor_code_execution(command=")
    assert analysis.repeated_indexes == {1}


# ---------------------------------------------------------------- view


def test_no_directive_returns_history_untouched() -> None:
    messages = _history(_same_script(3))

    view, tools_disabled = apply_loop_guard_view(messages, None)

    assert view is messages
    assert tools_disabled is False


def test_note_annotates_only_repeated_results_without_mutating_history() -> None:
    messages = _history(_same_script(3))
    originals = [m.content for m in messages]

    view, tools_disabled = apply_loop_guard_view(
        messages, LoopGuardDirective(level=LoopGuardLevel.NOTE)
    )

    tool_messages = [m for m in view if m.role == MessageRole.TOOL]
    assert REPEAT_NOTE not in (tool_messages[0].content or "")
    assert all(REPEAT_NOTE in (m.content or "") for m in tool_messages[1:])
    assert all(m.additional_kwargs.get("tool_call_id") for m in tool_messages)
    assert [m.content for m in messages] == originals
    assert tools_disabled is False


def test_flatten_turns_steps_into_one_user_work_log() -> None:
    messages = _history(_failed_creates(3), previous_turn=True)

    view, tools_disabled = apply_loop_guard_view(
        messages, LoopGuardDirective(level=LoopGuardLevel.FLATTEN)
    )

    # Earlier turns are kept; everything after the latest user message is folded.
    assert view[:4] == messages[:4]
    assert len(view) == 5
    work_log = view[-1]
    assert work_log.role == MessageRole.USER
    text = work_log.content or ""
    assert text.startswith("Compute the penalty")
    assert "<work_log>" in text
    assert text.count("action: text_editor_code_execution(") == 3
    assert text.count(_CREATE_ERROR) == 3
    assert text.count(REPEAT_NOTE) == 2
    assert FINAL_INSTRUCTIONS not in text
    assert tools_disabled is False


def test_final_disables_tools_and_asks_for_the_answer() -> None:
    view, tools_disabled = apply_loop_guard_view(
        _history(_same_script(5)), LoopGuardDirective(level=LoopGuardLevel.FINAL)
    )

    assert tools_disabled is True
    text = view[-1].content or ""
    assert text.endswith(FINAL_INSTRUCTIONS)
    assert "Claimable: 1273.80" in text


def test_flatten_truncates_large_arguments() -> None:
    big = {"command": "cat > f.csv << EOF\n" + "row\n" * 5000}
    steps: list[Step] = [("bash_code_execution", big, "File created at f.csv ok")] * 3

    view, _ = apply_loop_guard_view(
        _history(steps), LoopGuardDirective(level=LoopGuardLevel.FLATTEN)
    )

    assert len(view[-1].content or "") < 5000


def test_directive_without_tool_steps_keeps_history() -> None:
    messages = [ChatMessage(role=MessageRole.USER, content="hi")]

    view, tools_disabled = apply_loop_guard_view(
        messages, LoopGuardDirective(level=LoopGuardLevel.FINAL)
    )

    assert view is messages
    assert tools_disabled is False


# ---------------------------------------------------------------- interceptor


def _interceptor(**overrides: Any) -> RepeatGuardRequestInterceptor:
    settings = MagicMock()
    settings.chat.repeat_guard = RepeatGuardSettings(**overrides)
    return RepeatGuardRequestInterceptor(settings)


def _context(messages: list[ChatMessage]) -> SimpleNamespace:
    state = SimpleNamespace(
        input=SimpleNamespace(
            request=SimpleNamespace(messages=messages), loop_guard=None
        )
    )
    context = SimpleNamespace(phase=InterceptorPhase.BEFORE_ITERATION, state=state)
    context.set_state = lambda new_state: None
    return context


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (_same_script(1), None),
        (_same_script(2), LoopGuardLevel.NOTE),
        (_same_script(3), LoopGuardLevel.FLATTEN),
        (_same_script(4), LoopGuardLevel.FLATTEN),
        (_same_script(5), LoopGuardLevel.FINAL),
        (_failed_creates(3), LoopGuardLevel.FLATTEN),
        (_failed_creates(6), LoopGuardLevel.FINAL),
    ],
)
async def test_escalation_levels(
    steps: list[Step], expected: LoopGuardLevel | None
) -> None:
    context = _context(_history(steps))

    await _interceptor().intercept(context)

    directive = context.state.input.loop_guard
    assert (directive.level if directive else None) == expected


@pytest.mark.anyio
async def test_new_user_message_resets_escalation() -> None:
    messages = _history(_same_script(5))
    messages.append(ChatMessage(role=MessageRole.USER, content="thanks, now plot it"))
    context = _context(messages)

    await _interceptor().intercept(context)

    assert context.state.input.loop_guard is None


@pytest.mark.anyio
async def test_disabled_guard_does_nothing() -> None:
    context = _context(_history(_same_script(5)))

    await _interceptor(enabled=False).intercept(context)

    assert context.state.input.loop_guard is None


@pytest.mark.anyio
async def test_only_runs_before_iteration() -> None:
    context = _context(_history(_same_script(5)))
    context.phase = InterceptorPhase.AFTER_ITERATION

    await _interceptor().intercept(context)

    assert context.state.input.loop_guard is None
