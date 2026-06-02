from typing import Any

from jinja2 import Template
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.generic_utils import prompt_to_messages
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.types import BaseOutputParser


class RichPromptTemplate(BasePromptTemplate):
    """A prompt template that uses Jinja2 templates for formatting."""

    kwargs: dict[str, Any]

    _template: Template
    _template_source: str

    def __init__(
        self,
        template_str: str,
        template: Template,
        template_kwargs: dict[str, Any] | None = None,
        output_parser: BaseOutputParser | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metadata is None:
            metadata = {}

        super().__init__(
            kwargs=template_kwargs or {},
            template_vars=[],
            output_parser=output_parser,
            metadata=metadata,
            **kwargs,
        )

        self._template_source = template_str
        self._template = template

    def partial_format(self, **kwargs: Any) -> "RichPromptTemplate":
        """Partially format the prompt with the given kwargs."""
        output_parser = self.output_parser
        self.output_parser = None

        prompt = self
        prompt.kwargs.update(kwargs)

        prompt.output_parser = output_parser
        self.output_parser = output_parser

        return prompt

    def format(
        self,
        llm: BaseLLM | None = None,
        **kwargs: Any,
    ) -> str:
        """Format the prompt into a string."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }

        prompt = self._template.render(**all_kwargs)

        if self.output_parser is not None:
            prompt = self.output_parser.format(prompt)

        return prompt.strip() if prompt else ""

    def format_messages(
        self, llm: BaseLLM | None = None, **kwargs: Any
    ) -> list[ChatMessage]:
        """Format the prompt into a list of chat messages."""
        prompt = self.format(llm=llm, **kwargs)
        return prompt_to_messages(prompt)

    def get_template(self, llm: BaseLLM | None = None) -> str:
        """Return the template string."""
        return self._template_source
