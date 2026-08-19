from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import NO_TOOL_CONTENT
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    _resolve_active_skill_names,
    _resolve_skill_states,
)


def _tool_message(
    *,
    call_name: str,
    content: str | None,
    args: dict[str, str] | None = None,
) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content=content,
        additional_kwargs={
            "tool_call_name": call_name,
            "tool_call_args": args or {},
        },
    )


def test_resolve_skill_states_from_load_result_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


def test_resolve_skill_states_falls_back_to_tool_call_args() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content=NO_TOOL_CONTENT,
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


def test_resolve_skill_states_ignores_failed_load_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "error": "missing"}',
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == set()


def test_resolve_skill_states_unload_from_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        ),
        _tool_message(
            call_name=SKILL_UNLOAD_TOOL_NAME,
            content='{"name": "skill-creator", "unloaded": true}',
            args={"name": "skill-creator"},
        ),
    ]

    active, removed = _resolve_skill_states(messages)
    assert active == set()
    assert removed == {"skill-creator"}


def test_resolve_skill_states_unload_falls_back_to_tool_call_args() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        ),
        _tool_message(
            call_name=SKILL_UNLOAD_TOOL_NAME,
            content=NO_TOOL_CONTENT,
            args={"name": "skill-creator"},
        ),
    ]

    active, removed = _resolve_skill_states(messages)
    assert active == set()
    assert removed == {"skill-creator"}
