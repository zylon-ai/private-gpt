import pytest
from llama_index.llms import ChatMessage, MessageRole

from private_gpt.components.llm.prompt.prompt_helper import (
    DefaultPromptStyle,
    LlamaIndexPromptStyle,
    VigognePromptStyle,
    get_prompt_style,
)


@pytest.mark.parametrize(
    ("prompt_style", "expected_prompt_style"),
    [
        ("default", DefaultPromptStyle),
        ("llama2", LlamaIndexPromptStyle),
        ("tag", VigognePromptStyle),
    ],
)
def test_get_prompt_style_success(prompt_style, expected_prompt_style):
    assert get_prompt_style(prompt_style) == expected_prompt_style


def test_get_prompt_style_failure():
    prompt_style = "unknown"
    with pytest.raises(ValueError) as exc_info:
        get_prompt_style(prompt_style)
    assert str(exc_info.value) == f"Unknown prompt_style='{prompt_style}'"


def test_tag_prompt_style_format():
    prompt_style = VigognePromptStyle()
    messages = [
        ChatMessage(content="You are an AI assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<|system|>: You are an AI assistant.\n"
        "<|user|>: Hello, how are you doing?\n"
        "<|assistant|>: "
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt


def test_tag_prompt_style_format_with_system_prompt():
    system_prompt = "This is a system prompt from configuration."
    prompt_style = VigognePromptStyle(default_system_prompt=system_prompt)
    messages = [
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        f"<|system|>: {system_prompt}\n"
        "<|user|>: Hello, how are you doing?\n"
        "<|assistant|>: "
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt

    messages = [
        ChatMessage(
            content="FOO BAR Custom sys prompt from messages.", role=MessageRole.SYSTEM
        ),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<|system|>: FOO BAR Custom sys prompt from messages.\n"
        "<|user|>: Hello, how are you doing?\n"
        "<|assistant|>: "
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt


def test_llama2_prompt_style_format():
    prompt_style = LlamaIndexPromptStyle()
    messages = [
        ChatMessage(content="You are an AI assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<s> [INST] <<SYS>>\n"
        " You are an AI assistant. \n"
        "<</SYS>>\n"
        "\n"
        " Hello, how are you doing? [/INST]"
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt


def test_llama2_prompt_style_with_system_prompt():
    system_prompt = "This is a system prompt from configuration."
    prompt_style = LlamaIndexPromptStyle(default_system_prompt=system_prompt)
    messages = [
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<s> [INST] <<SYS>>\n"
        f" {system_prompt} \n"
        "<</SYS>>\n"
        "\n"
        " Hello, how are you doing? [/INST]"
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt

    messages = [
        ChatMessage(
            content="FOO BAR Custom sys prompt from messages.", role=MessageRole.SYSTEM
        ),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<s> [INST] <<SYS>>\n"
        " FOO BAR Custom sys prompt from messages. \n"
        "<</SYS>>\n"
        "\n"
        " Hello, how are you doing? [/INST]"
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt
