import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from llama_index.llms import ChatMessage, MessageRole

try:
    from private_gpt.components.llm.prompt.prompt_helper import (
        DefaultPromptStyle,
        LlamaCppPromptStyle,
        LlamaIndexPromptStyle,
        TemplatePromptStyle,
        VigognePromptStyle,
        get_prompt_style,
    )
except ImportError:
    DefaultPromptStyle = None
    LlamaCppPromptStyle = None
    LlamaIndexPromptStyle = None
    TemplatePromptStyle = None
    VigognePromptStyle = None
    get_prompt_style = None


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
@pytest.mark.parametrize(
    ("prompt_style", "expected_prompt_style"),
    [
        (None, DefaultPromptStyle),
        ("llama2", LlamaIndexPromptStyle),
        ("vigogne", VigognePromptStyle),
        ("llama_cpp.alpaca", LlamaCppPromptStyle),
        ("llama_cpp.zephyr", LlamaCppPromptStyle),
    ],
)
def test_get_prompt_style_success(prompt_style, expected_prompt_style):
    assert type(get_prompt_style(prompt_style)) == expected_prompt_style


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
def test_get_prompt_style_template_success():
    jinja_template = "{% for message in messages %}<|{{message['role']}}|>: {{message['content'].strip() + '\\n'}}{% endfor %}<|assistant|>: "
    with NamedTemporaryFile("w") as tmp_file:
        path = Path(tmp_file.name)
        tmp_file.write(jinja_template)
        tmp_file.flush()
        tmp_file.seek(0)
        prompt_style = get_prompt_style(
            "template", template_name=path.name, template_dir=path.parent
        )
        assert type(prompt_style) == TemplatePromptStyle
        prompt = prompt_style.messages_to_prompt(
            [
                ChatMessage(
                    content="You are an AI assistant.", role=MessageRole.SYSTEM
                ),
                ChatMessage(content="Hello, how are you doing?", role=MessageRole.USER),
            ]
        )

        expected_prompt = (
            "<|system|>: You are an AI assistant.\n"
            "<|user|>: Hello, how are you doing?\n"
            "<|assistant|>: "
        )

        assert prompt == expected_prompt


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
def test_get_prompt_style_failure():
    prompt_style = "unknown"
    with pytest.raises(ValueError) as exc_info:
        get_prompt_style(prompt_style)
    assert str(exc_info.value) == f"Unknown prompt_style='{prompt_style}'"


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
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


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
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


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
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


@pytest.mark.skipif(
    "llama_cpp" not in sys.modules, reason="requires the llama-cpp-python library"
)
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
