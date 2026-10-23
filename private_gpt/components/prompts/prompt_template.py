from pathlib import Path
from typing import Any

from injector import singleton
from jinja2 import Environment, FileSystemLoader, Template
from llama_index.core import BasePromptTemplate

from private_gpt.components.prompts.rich_template import RichPromptTemplate
from private_gpt.paths import prompt_templates_path


@singleton
class PromptTemplateService:
    def __init__(self, templates_dir: Path = prompt_templates_path) -> None:
        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do"],
        )

    def get_template(self, template_name: str) -> Template:
        return self.env.get_template(template_name)

    def get_template_str(self, template_name: str) -> str:
        """Get the template string from the template name."""
        template = self.get_template(template_name)
        if hasattr(template, "source"):
            return str(template.source)
        elif hasattr(template, "filename") and template.filename:
            template_path = Path(template.filename)
            with open(template_path, encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(
                f"Template {template_name} does not have a source or filename."
            )

    def create_prompt_template(
        self,
        template_name: str,
        **template_kwargs: Any,
    ) -> BasePromptTemplate:
        template = self.get_template(template_name)
        template_str = self.get_template_str(template_name)

        def process_value(v: Any) -> Any:
            if isinstance(v, list):
                return [process_value(value) for value in v]
            elif isinstance(v, dict):
                return {
                    process_value(key): process_value(val) for key, val in v.items()
                }
            else:
                return str(v)

        template_kwargs = {
            k: process_value(v) for k, v in template_kwargs.items() if k and v
        }
        return RichPromptTemplate(
            template=template,
            template_str=template_str,
            template_kwargs=template_kwargs,
        )

    def concat_prompts(
        self,
        *prompts: BasePromptTemplate,
        separator: str = "\n",
    ) -> BasePromptTemplate:
        """Concatenate multiple prompts into a single prompt."""
        if not prompts or len(prompts) == 0:
            return RichPromptTemplate(template=Template(""), template_str="")
        if len(prompts) == 1:
            return prompts[0]

        templates = [
            prompt.get_template() for prompt in prompts if prompt.get_template()
        ]
        template_kwargs = {}
        for prompt in prompts:
            template_kwargs.update(prompt.kwargs)

        template_str = separator.join(templates)
        template = self.env.from_string(template_str)
        return RichPromptTemplate(
            template=template,
            template_str=template_str,
            template_kwargs=template_kwargs,
        )
