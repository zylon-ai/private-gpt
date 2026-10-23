from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.llm.prompt_styles.chat_template_prompt_style import (
    ChatTemplatePromptStyle,
    _serialize_tool_calls,
)


def test_sanitize_conversation_keeps_tool_results_as_user_tool_response() -> None:
    sanitized = ChatTemplatePromptStyle._sanitize_conversation(
        [
            {"role": "user", "content": "Create a skill"},
            {"role": "assistant", "content": "Loading skill-creator"},
            {
                "role": "tool",
                "content": '{"name": "skill-creator", "loaded": true}',
                "tool_call_id": "srvtoolu_load",
            },
        ]
    )

    assert sanitized[-1]["role"] == "user"
    assert "<tool_response>" in sanitized[-1]["content"]
    assert "skill-creator" in sanitized[-1]["content"]


def test_serialize_tool_calls_converts_tool_selection() -> None:
    serialized = _serialize_tool_calls(
        [
            ToolSelection(
                tool_id="srvtoolu_load_skill_1",
                tool_name="load_skill",
                tool_kwargs={"name": "skill-creator"},
            )
        ]
    )

    assert serialized == [
        {
            "id": "srvtoolu_load_skill_1",
            "type": "function",
            "function": {
                "name": "load_skill",
                "arguments": '{"name": "skill-creator"}',
            },
        }
    ]


def test_to_hf_messages_does_not_spread_non_serializable_kwargs() -> None:
    style = ChatTemplatePromptStyle(
        tokenizer=SimpleNamespace(apply_chat_template=lambda **_: "")
    )
    messages = [
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"name": "skill-creator", "loaded": true}',
            additional_kwargs={
                "tool_call_id": "srvtoolu_load",
                "tool_call_name": "load_skill",
                "server_tool_result": [object()],
            },
        )
    ]

    hf = style._to_hf_messages(messages)

    assert hf == [
        {
            "role": "tool",
            "content": '{"name": "skill-creator", "loaded": true}',
            "tool_call_id": "srvtoolu_load",
            "name": "load_skill",
        }
    ]
