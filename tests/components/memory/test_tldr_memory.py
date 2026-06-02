import pytest
from llama_index.core.base.llms.types import ChatMessage

from private_gpt.components.chat.processors.chat_history.memory.tldr_utils import (
    trim_to_last_tldr,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _left(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content, additional_kwargs={"tldr": "left"})


def _right(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content, additional_kwargs={"tldr": "right"})


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content, additional_kwargs={})


def _roles(messages: list[ChatMessage]) -> list[str]:
    return [str(m.role.value) for m in messages]


# ---------------------------------------------------------------------------
# Left TLDR tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chat_history", "expected_roles"),
    [
        # Empty input
        ([], []),
        # No TLDRs — unchanged
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "q2"),
                _msg("assistant", "a2"),
            ],
            ["user", "assistant", "user", "assistant"],
        ),
        # Left TLDR starts with a user message → drop everything before it
        (
            [
                _msg("user", "old"),
                _msg("assistant", "old-reply"),
                _left("user", "summary"),
                _left("assistant", "summary"),
                _msg("assistant", "extra"),
                _msg("user", "new"),
                _msg("assistant", "new-reply"),
            ],
            ["user", "assistant", "assistant", "user", "assistant"],
        ),
        # Last contiguous left-TLDR block is assistant-starting;
        # earlier non-contiguous block in same group is dropped, tail is kept
        (
            [
                _msg("user", "q1"),  # keep (different group)
                _msg("assistant", "a1"),  # keep
                _msg("user", "q2"),  # keep (group user start)
                _left("assistant", "s"),  # drop (not last contiguous run)
                _left("assistant", "s"),  # drop
                _msg("assistant", "extra"),  # drop (between runs)
                _left("assistant", "s"),  # keep (last contiguous run starts here)
                _left("assistant", "s"),  # keep
                _left("assistant", "s"),  # keep
                _msg("assistant", "tail"),  # keep (after last run)
            ],
            [
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
                "assistant",
            ],
        ),
        # Mixed: user-starting left TLDR earlier, assistant-starting left TLDR last;
        # messages between history_start and group_user_start+1 are kept,
        # junk before the last run within the group is dropped
        (
            [
                _msg("user", "old"),
                _msg("assistant", "old-reply"),
                _left("user", "summary"),  # history_start — drop everything before
                _left("assistant", "summary"),
                _msg("assistant", "extra"),
                _msg("user", "q"),  # group_user_start
                _msg("assistant", "junk"),  # dropped (before last run)
                _left("assistant", "s"),  # last contiguous run
                _msg("assistant", "tail"),  # kept (after last run)
            ],
            ["user", "assistant", "assistant", "user", "assistant", "assistant"],
        ),
    ],
)
def test_trim_left_tldr(
    chat_history: list[ChatMessage], expected_roles: list[str]
) -> None:
    assert _roles(trim_to_last_tldr(chat_history)) == expected_roles


# ---------------------------------------------------------------------------
# Right TLDR tests
#
# Invariant: user messages are ALWAYS kept.
# Per user-group: find last contiguous right-TLDR run → keep user + that run only.
# Everything else in the group (before and after the run) is dropped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chat_history", "expected_roles"),
    [
        # Right TLDR in middle of group → keep user + last run only (more is dropped)
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "q2"),  # kept (user invariant)
                _msg("assistant", "partial"),  # dropped (before last run)
                _right("assistant", "summary"),  # kept (last contiguous run)
                _msg("assistant", "more"),  # dkeep
                _msg("user", "q3"),
                _msg("assistant", "a3"),
            ],
            [
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant",
                "user",
                "assistant",
            ],
        ),
        # Right TLDR is last message in group before next user →
        # keep user + summary (partial dropped)
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "q2"),  # kept
                _msg("assistant", "partial"),  # dropped
                _right("assistant", "summary"),  # kept (last contiguous run)
                _msg("user", "q3"),
                _msg("assistant", "a3"),
            ],
            ["user", "assistant", "user", "assistant", "user", "assistant"],
        ),
        # Right TLDR is the only message in group → keep user + summary
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "q2"),  # kept
                _right("assistant", "summary"),  # kept (only msg, last contiguous run)
                _msg("user", "q3"),
                _msg("assistant", "a3"),
            ],
            ["user", "assistant", "user", "assistant", "user", "assistant"],
        ),
        # Right TLDR last in history (no next user) → keep user + summary
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "q2"),  # kept
                _msg("assistant", "partial"),  # dropped
                _right(
                    "assistant", "summary"
                ),  # kept (last contiguous run, end of history)
            ],
            ["user", "assistant", "user", "assistant"],
        ),
        # Right TLDR only message, last in history → keep user + summary
        (
            [
                _msg("user", "q1"),
                _right("assistant", "summary"),
            ],
            ["user", "assistant"],
        ),
        # Multiple consecutive right TLDRs → keep user + last contiguous run (s1+s2),
        # drop partial before and more after
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "partial"),  # dropped
                _msg("assistant", "partial"),  # dropped
                _right("assistant", "s1"),  # kept (last contiguous run starts here)
                _right("assistant", "s2"),  # kept
                _msg("assistant", "more"),  # kept
                _msg("user", "q2"),
                _msg("assistant", "a2"),
            ],
            ["user", "assistant", "assistant", "assistant", "user", "assistant"],
        ),
        # Multiple non-consecutive right TLDRs → last contiguous run is the final
        # single right TLDR; everything before it dropped
        (
            [
                _msg("user", "q1"),
                _msg("assistant", "partial"),  # dropped
                _msg("assistant", "partial"),  # dropped
                _right("assistant", "s1"),  # dropped (not last contiguous run)
                _right("assistant", "s2"),  # dropped
                _msg("assistant", "more"),  # dropped
                _right("assistant", "s3"),  # kept (last contiguous run)
                _msg("user", "q2"),
                _msg("assistant", "a2"),
            ],
            ["user", "assistant", "user", "assistant"],
        ),
        # Group with no user message (preamble) → left unchanged
        (
            [
                _msg("assistant", "preamble"),
                _right("assistant", "summary"),
                _msg("user", "q1"),
                _msg("assistant", "a1"),
            ],
            ["assistant", "assistant", "user", "assistant"],
        ),
    ],
)
def test_trim_right_tldr(
    chat_history: list[ChatMessage], expected_roles: list[str]
) -> None:
    assert _roles(trim_to_last_tldr(chat_history)) == expected_roles


# ---------------------------------------------------------------------------
# Combined left + right TLDR tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chat_history", "expected_roles"),
    [
        # Right trim: q1 group keeps user(q1)+summary (partial+more dropped).
        # Left trim: condensed(user) is history_start → drops old+old-reply.
        (
            [
                _msg("user", "old"),
                _msg("assistant", "old-reply"),
                _left("user", "condensed"),  # history_start
                _left("assistant", "condensed"),
                _msg("assistant", "extra"),
                _msg("user", "q1"),  # kept (user invariant)
                _msg("assistant", "partial"),  # dropped by right trim
                _right("assistant", "summary"),  # kept (last run)
                _msg("assistant", "more"),  # kept
                _msg("user", "q2"),
                _msg("assistant", "a2"),
            ],
            # condensed(L-user), condensed(L-asst), extra, q1, summary, q2, a2
            [
                "user",
                "assistant",
                "assistant",
                "user",
                "assistant",
                "assistant",
                "user",
                "assistant",
            ],
        ),
        # Right trim: q1 group keeps user(q1)+summary.
        # Left trim: condensed(user) is history_start → drops very-old.
        (
            [
                _msg("user", "very-old"),
                _left("user", "condensed"),  # history_start — drops very-old
                _left("assistant", "condensed"),
                _msg("user", "q1"),  # kept (user invariant)
                _msg("assistant", "partial"),
                _right("assistant", "summary1"),  # kept (only right, last run)
                _msg("assistant", "partial"),
                _right("assistant", "summary2"),  # kept (only right, last run)
                _msg("user", "q2"),
                _msg("assistant", "a2"),
            ],
            # condensed(L-user), condensed(L-asst), q1, summary2, q2, a2
            ["user", "assistant", "user", "assistant", "user", "assistant"],
        ),
    ],
)
def test_trim_combined_tldr(
    chat_history: list[ChatMessage], expected_roles: list[str]
) -> None:
    assert _roles(trim_to_last_tldr(chat_history)) == expected_roles
