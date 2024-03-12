import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from private_gpt.components.llm.prompt_helper import (
    ChatMLPromptStyle,
    DefaultPromptStyle,
    Llama2PromptStyle,
    MistralPromptStyle,
    TagPromptStyle,
    get_prompt_style,
)


@pytest.mark.parametrize(
    ("prompt_style", "expected_prompt_style"),
    [
        ("default", DefaultPromptStyle),
        ("llama2", Llama2PromptStyle),
        ("tag", TagPromptStyle),
        ("mistral", MistralPromptStyle),
        ("chatml", ChatMLPromptStyle),
    ],
)
def test_get_prompt_style_success(prompt_style, expected_prompt_style):
    assert isinstance(get_prompt_style(prompt_style), expected_prompt_style)


def test_get_prompt_style_failure():
    prompt_style = "unknown"
    with pytest.raises(ValueError) as exc_info:
        get_prompt_style(prompt_style)
    assert str(exc_info.value) == f"Unknown prompt_style='{prompt_style}'"


def test_tag_prompt_style_format():
    prompt_style = TagPromptStyle()
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
    prompt_style = TagPromptStyle()
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


def test_mistral_prompt_style_format():
    prompt_style = MistralPromptStyle()
    messages = [
        ChatMessage(content="You are an AI assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<s>[INST] You are an AI assistant. [/INST]</s>"
        "[INST] Hello, how are you doing? [/INST]"
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt


def test_chatml_prompt_style_format():
    prompt_style = ChatMLPromptStyle()
    messages = [
        ChatMessage(content="You are an AI assistant.", role=MessageRole.SYSTEM),
        ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
    ]

    expected_prompt = (
        "<|im_start|>system\n"
        "You are an AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello, how are you doing?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    assert prompt_style.messages_to_prompt(messages) == expected_prompt


def test_llama2_prompt_style_format():
    prompt_style = Llama2PromptStyle()
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
    prompt_style = Llama2PromptStyle()
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
