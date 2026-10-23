import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from jinja2 import Template
from llama_index.core import BasePromptTemplate
from llama_index.core.llms import ChatMessage

from private_gpt.components.prompts.rich_template import RichPromptTemplate


@pytest.fixture
def templates_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        with open(dir_path / "simple.j2", "w") as f:
            f.write("Hello {{ name }}!")

        with open(dir_path / "complex.j2", "w") as f:
            f.write(
                """
            # {{ title }}

            {{ content }}

            {% if footer %}
            ---
            {{ footer }}
            {% endif %}
            """
            )

        yield dir_path


@pytest.fixture
def prompt_service(templates_dir: Path) -> Any:
    from private_gpt.components.prompts.prompt_template import PromptTemplateService

    return PromptTemplateService(templates_dir=templates_dir)


def test_rich_prompt_template_string_init() -> None:
    template_str = "Hello {{ name }}!"
    template = Template(template_str)
    prompt = RichPromptTemplate(template_str=template_str, template=template)

    assert prompt.get_template() == template_str
    result = prompt.format(name="World")
    assert result == "Hello World!"


def test_rich_prompt_template_template_init() -> None:
    template_str = "Hello {{ name }}!"
    template = Template(template_str)
    prompt = RichPromptTemplate(template_str=template_str, template=template)

    result = prompt.format(name="World")
    assert result == "Hello World!"


def test_rich_prompt_template_partial_format() -> None:
    template_str = "Hello {{ first_name }} {{ last_name }}!"
    template = Template(template_str)
    prompt = RichPromptTemplate(template_str=template_str, template=template)

    partial_prompt = prompt.partial_format(first_name="John")

    assert prompt.format(first_name="Jane", last_name="Doe") == "Hello Jane Doe!"
    assert partial_prompt.format(last_name="Smith") == "Hello John Smith!"


def test_rich_prompt_template_format_messages() -> None:
    template_str = "Hello {{ name }}!"
    template = Template(template_str)
    prompt = RichPromptTemplate(template_str=template_str, template=template)

    messages = prompt.format_messages(name="World")
    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    assert messages[0].content == "Hello World!"


def test_prompt_template_service_get_template(prompt_service: Any) -> None:
    template = prompt_service.get_template("simple.j2")
    assert template is not None
    assert template.render(name="World") == "Hello World!"


def test_prompt_template_service_create_prompt_template(prompt_service: Any) -> None:
    prompt = prompt_service.create_prompt_template("simple.j2", name="World")

    assert isinstance(prompt, BasePromptTemplate)
    assert prompt.format() == "Hello World!"


@pytest.mark.parametrize(
    ("template_name", "template_kwargs", "expected_output"),
    [
        ("simple.j2", {"name": "World"}, "Hello World!"),
        (
            "complex.j2",
            {"title": "Test", "content": "Some content", "footer": "Footer text"},
            "\n            # Test\n\n            Some content\n\n            ---\n            Footer text\n            ",
        ),
        (
            "complex.j2",
            {"title": "Test", "content": "Some content"},
            "\n            # Test\n\n            Some content\n\n            ",
        ),
    ],
)
def test_prompt_template_service_create_prompt_template_parameterized(
    prompt_service: Any,
    template_name: str,
    template_kwargs: dict[str, str],
    expected_output: str,
) -> None:
    prompt = prompt_service.create_prompt_template(template_name, **template_kwargs)
    assert prompt.format().strip() == expected_output.strip()


def test_prompt_template_service_concat_prompts_empty(prompt_service: Any) -> None:
    prompt = prompt_service.concat_prompts()
    assert isinstance(prompt, BasePromptTemplate)
    assert prompt.format() == ""


def test_prompt_template_service_concat_prompts_single(prompt_service: Any) -> None:
    original_prompt = prompt_service.create_prompt_template("simple.j2", name="World")
    prompt = prompt_service.concat_prompts(original_prompt)

    assert prompt == original_prompt
    assert prompt.format() == "Hello World!"


def test_prompt_template_service_concat_prompts_multiple(prompt_service: Any) -> None:
    template_str1 = "Hello {{ name }}!"
    template1 = Template(template_str1)
    prompt1 = RichPromptTemplate(template_str=template_str1, template=template1)

    template_str2 = "How are you, {{ name }}?"
    template2 = Template(template_str2)
    prompt2 = RichPromptTemplate(template_str=template_str2, template=template2)

    combined = prompt_service.concat_prompts(prompt1, prompt2)

    assert combined.get_template() == "Hello {{ name }}!\nHow are you, {{ name }}?"
    result = combined.format(name="World")
    assert result == "Hello World!\nHow are you, World?"


def test_get_template_returns_source_after_concat(prompt_service: Any) -> None:
    prompt1 = prompt_service.create_prompt_template("simple.j2")
    prompt2 = prompt_service.create_prompt_template("complex.j2")

    combined = prompt_service.concat_prompts(prompt1, prompt2)

    template_str = combined.get_template()
    assert "Hello" in template_str
    assert "title" in template_str
    assert "content" in template_str
