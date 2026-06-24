import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import injector
from injector import singleton
from jinja2 import TemplateNotFound
from llama_index.core import BasePromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore

from private_gpt.components.engines.citations.format import (
    format_context,
    format_llm_source,
)
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.components.prompts.prompt_template import PromptTemplateService
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec

logger = logging.getLogger(__name__)


class _ToolNamespace:
    """Lightweight namespace for per-tool Jinja2 templates."""

    __slots__ = ("tools",)

    def __init__(self, tools: dict[str, Any]) -> None:
        self.tools = tools


def _build_tool_namespace(tools: list["ToolSpec"]) -> _ToolNamespace:
    ns: dict[str, Any] = {}
    for tool in tools:
        if tool.name is None:
            continue
        try:
            canonical = tool.get_original_tool_name()
        except ValueError:
            canonical = tool.name
        ns[f"{canonical}_tool_name"] = tool.name
        ns[f"has_{canonical}"] = True
        if tool.name != canonical:
            ns[f"{tool.name}_tool_name"] = tool.name
            ns[f"has_{tool.name}"] = True
    return _ToolNamespace(ns)


@singleton
class PromptBuilderService:
    @injector.inject
    def __init__(self, prompt_template_service: PromptTemplateService) -> None:
        self.template_service = prompt_template_service

    def create_context_prompt(
        self,
        documents: list[Document] | None = None,
        nodes: list[NodeWithScore] | None = None,
        generate_citations: bool = False,
        token_limit: int | None = None,
        tokenizer_fn: TokenizerFn | None = None,
        included_in_system_prompt: bool = False,
    ) -> tuple[BasePromptTemplate, list[Document] | None]:
        if documents is None and nodes is not None:
            documents = [Document.from_node(node) for node in nodes]

        if not documents:
            return PromptTemplate(template=""), None

        documents, context_str = format_context(
            documents=documents,
            generate_citations=generate_citations,
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn,
        )

        prompt = self.template_service.create_prompt_template(
            "context/context.j2",
            context_str=context_str,
            included_in_system_prompt=included_in_system_prompt,
        )
        return prompt, documents

    def create_chat_condense_prompt(
        self,
        question: str,
        chat_history: str,
        max_words: int | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "chat/condense/condense.j2",
            question=question,
            chat_history=chat_history,
            max_words=max_words,
            few_shots=few_shots,
        )

    def create_chat_header_prompt(
        self,
        assistant_name: str | None = None,
        assistant_description: str | None = None,
        system_prompt: str | None = None,
        current_date: datetime | None = None,
    ) -> BasePromptTemplate:
        if not current_date:
            current_date = datetime.now()
        return self.template_service.create_prompt_template(
            "chat/system/base.j2",
            system_prompt=None,
            assistant_name=assistant_name or settings().chat.assistant_name,
            assistant_description=assistant_description
            or settings().chat.assistant_description,
            # Use date-level granularity: a sub-second timestamp here changes on
            # every request and sits at the front of the system prompt (before the
            # guidelines and retrieved context), which defeats LLM prompt-prefix
            # caching (e.g. OpenAI automatic prefix caching, local KV-cache reuse).
            # The model only needs the calendar date for relative-date reasoning.
            current_date=current_date.astimezone().date().isoformat(),
        )

    def create_chat_context_for_system_prompt(
        self,
        documents: list[Document] | None,
        generate_citations: bool,
        guidelines_prompt: BasePromptTemplate | None = None,
        token_limit: int | None = None,
        tokenizer_fn: TokenizerFn | None = None,
    ) -> tuple[BasePromptTemplate | None, list[Document] | None]:
        if not documents:
            return None, documents

        available_tokens: int | None = None
        if token_limit is not None and tokenizer_fn is not None:
            guidelines_text = guidelines_prompt.format() if guidelines_prompt else ""
            base_prompt_length = len(tokenizer_fn(guidelines_text))
            available_tokens = max(0, token_limit - base_prompt_length)

        context_prompt, filtered_documents = self.create_context_prompt(
            documents=documents,
            generate_citations=generate_citations,
            token_limit=available_tokens,
            tokenizer_fn=tokenizer_fn,
            included_in_system_prompt=True,
        )
        if not context_prompt.get_template():
            return None, filtered_documents
        return context_prompt, filtered_documents

    def create_summary_prompt(
        self,
        system_prompt: str | PromptTemplate | None = None,
        user_query: str | None = None,
        additional_instructions: str | None = None,
        max_words: int | None = None,
        add_rules: bool = True,
    ) -> BasePromptTemplate:
        if not system_prompt:
            system_prompt = self.template_service.create_prompt_template(
                "summary/base.j2",
            ).format()

        rules = (
            self.template_service.create_prompt_template(
                "summary/rules.j2",
            ).format()
            if add_rules
            else None
        )

        return self.template_service.create_prompt_template(
            "summary/summary.j2",
            system_prompt=system_prompt,
            rules=rules,
            user_query=user_query,
            additional_instructions=additional_instructions,
            max_words=max_words,
        )

    def create_summary_history_in_details(
        self,
        user_query: str,
        chat_history: list[ChatMessage] | None = None,
        system_prompt: str | PromptTemplate | None = None,
        max_words: int | None = None,
        aggressive_level: int | None = None,
        few_shots: bool = True,
        messages_to_history_str_fn: Callable[[Sequence[ChatMessage]], str]
        | None = None,
    ) -> BasePromptTemplate:
        if chat_history is not None and len(chat_history) == 0:
            return PromptTemplate(template="")

        if system_prompt and isinstance(system_prompt, PromptTemplate):
            system_prompt = system_prompt.format()

        if messages_to_history_str_fn is None:
            from llama_index.core.base.llms.generic_utils import messages_to_history_str

            messages_to_history_str_fn = messages_to_history_str

        chat_history_str = (
            messages_to_history_str_fn(chat_history)
            if chat_history is not None
            else None
        )
        return self.template_service.create_prompt_template(
            "memory/tldr/in_detail.j2",
            system_prompt=system_prompt,
            query_str=user_query,
            chat_history=chat_history_str,
            max_words=max_words,
            few_shots=few_shots,
            aggressive_level=aggressive_level,
        )

    def create_summary_history_approximately(
        self,
        chat_history: list[ChatMessage] | None = None,
        system_prompt: str | PromptTemplate | None = None,
        max_words: int | None = None,
        few_shots: bool = True,
        messages_to_history_str_fn: Callable[[Sequence[ChatMessage]], str]
        | None = None,
    ) -> BasePromptTemplate:
        if chat_history is not None and len(chat_history) == 0:
            return PromptTemplate(template="")

        if system_prompt and isinstance(system_prompt, PromptTemplate):
            system_prompt = system_prompt.format()

        if messages_to_history_str_fn is None:
            from llama_index.core.base.llms.generic_utils import messages_to_history_str

            messages_to_history_str_fn = messages_to_history_str

        chat_history_str = (
            messages_to_history_str_fn(chat_history)
            if chat_history is not None
            else None
        )
        return self.template_service.create_prompt_template(
            "memory/tldr/history_approxymately.j2",
            system_prompt=system_prompt,
            text=chat_history_str,
            max_words=max_words,
            few_shots=few_shots,
        )

    def create_report_generation_section_prompt(
        self,
        query_str: str | None = None,
        documents: list[Document] | None = None,
        nodes: list[NodeWithScore] | None = None,
        section_separator: str = "END_OF_SECTION",
        empty_section: str = "EMPTY_SECTION",
        token_limit: int | None = None,
        tokenizer_fn: TokenizerFn | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        if documents is None and nodes is not None:
            documents = [Document.from_node(node) for node in nodes]
        _, context_str = format_context(
            documents=documents,
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn,
            generate_citations=False,
        )
        return self.template_service.create_prompt_template(
            "report/section/generation.j2",
            query_str=query_str,
            context_str=context_str,
            section_separator=section_separator,
            section_empty=empty_section,
            few_shots=few_shots,
        )

    def create_report_refine_section_prompt(
        self,
        query_str: str,
        context_str: str,
        existing_answer: str | None = None,
        additional_instructions: str | None = None,
        section_separator: str = "END_OF_SECTION",
        empty_section: str = "EMPTY_SECTION",
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "report/section/refine.j2",
            query_str=query_str,
            additional_instructions=additional_instructions,
            existing_answer=existing_answer,
            context_str=context_str,
            section_separator=section_separator,
            section_empty=empty_section,
        )

    def create_report_generation_content_prompt(
        self,
        query_str: str | None = None,
        empty_section: str = "EMPTY_SECTION",
        max_words: int | None = None,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "report/content/generation.j2",
            query_str=query_str,
            max_words=max_words,
            section_empty=empty_section,
        )

    def create_report_refine_content_prompt(
        self,
        query_str: str | None = None,
        empty_section: str = "EMPTY_SECTION",
        max_words: int | None = None,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "report/content/refine.j2",
            query_str=query_str,
            max_words=max_words,
            section_empty=empty_section,
        )

    def create_get_image_complexity_prompt(
        self,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/complexity.j2",
        )

    def create_image_interpretation_prompt(
        self,
        user_query: str | None = None,
        last_content: str | None = None,
        extraction_type: str | None = None,
        confidence: float | None = None,
        language: str | None = None,
        has_red_box: bool = False,
        has_structure: bool | None = None,
        errors: list[str] | None = None,
        suggestions: list[str] | None = None,
        max_words: int | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/interpretation.j2",
            user_query=user_query,
            last_content=last_content,
            extraction_type=extraction_type,
            confidence=confidence,
            language=language,
            has_red_box=has_red_box,
            has_structure=has_structure,
            errors=errors if errors else None,
            suggestions=suggestions if suggestions else None,
            max_words=max_words,
            few_shots=few_shots,
        )

    def create_image_interpretation_response(
        self,
        user_query: str | None = None,
        content: str | None = None,
        extraction_type: str | None = None,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/interpretation_response.j2",
            user_query=user_query,
            content=content,
            extraction_type=extraction_type,
        )

    def create_pptx_slide_fusion_prompt(
        self,
        extracted_text: str | None = None,
        max_words: int | None = None,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/pptx_slide_fusion.j2",
            extracted_text=extracted_text,
            max_words=max_words,
            has_text=extracted_text is not None and extracted_text != "",
        )

    def create_document_image_extract_prompt(
        self, max_words: int | None = None
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/document_image_extract.j2",
            max_words=max_words,
        )

    def create_image_strategy_prompt(
        self,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/classification.j2",
        )

    def create_image_evaluation_prompt(
        self,
        extraction_type: str | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        return self.template_service.create_prompt_template(
            "multimodality/images/evaluation.j2",
            extraction_type=extraction_type,
            few_shots=few_shots,
        )

    def create_audio_strategy_prompt(
        self,
    ) -> BasePromptTemplate:
        """Create prompt for inferring audio transcription strategy."""
        return self.template_service.create_prompt_template(
            "multimodality/audios/classification.j2",
        )

    def create_audio_transcription_prompt(
        self,
        user_query: str | None = None,
        last_content: str | None = None,
        audio_type: str | None = None,
        confidence: float | None = None,
        language: str | None = None,
        has_multiple_speakers: bool | None = None,
        enable_speaker_diarization: bool = False,
        errors: list[str] | None = None,
        suggestions: list[str] | None = None,
        max_words: int | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        """Create prompt for audio transcription with context."""
        return self.template_service.create_prompt_template(
            "multimodality/audios/transcription.j2",
            user_query=user_query,
            last_content=last_content,
            audio_type=audio_type,
            confidence=confidence,
            language=language,
            has_multiple_speakers=has_multiple_speakers,
            enable_speaker_diarization=enable_speaker_diarization,
            errors=errors if errors else None,
            suggestions=suggestions if suggestions else None,
            max_words=max_words,
            few_shots=few_shots,
        )

    def create_audio_transcription_response(
        self,
        user_query: str | None = None,
        transcript: str | None = None,
        speakers: list[str] | None = None,
        audio_type: str | None = None,
    ) -> BasePromptTemplate:
        """Create formatted response for audio transcription."""
        return self.template_service.create_prompt_template(
            "multimodality/audios/transcription_response.j2",
            user_query=user_query,
            transcript=transcript,
            speakers=speakers,
            audio_type=audio_type,
        )

    def create_audio_evaluation_prompt(
        self,
    ) -> BasePromptTemplate:
        """Create prompt for evaluating transcription quality."""
        return self.template_service.create_prompt_template(
            "multimodality/audios/evaluation.j2",
        )

    def create_audio_complexity_prompt(
        self,
    ) -> BasePromptTemplate:
        """Create prompt for assessing audio complexity."""
        return self.template_service.create_prompt_template(
            "multimodality/audios/complexity.j2",
        )

    def create_tool_instructions(
        self,
        canonical_name: str,
        tool_namespace: Any,
        few_shots: bool = False,
    ) -> BasePromptTemplate:
        """Create a per-tool instruction prompt for the given canonical tool name.

        Returns an empty ``PromptTemplate`` when no template exists for
        *canonical_name* or when rendering fails.

        :param canonical_name: Canonical tool name (e.g. ``"web_search"``).
        :param tool_namespace: Namespace object with a ``tools`` dict consumed by
            the template (e.g. ``{"web_search_tool_name": "online_search", ...}``).
        :param few_shots: Include few-shot examples in the rendered output.
        """
        template_path = f"chat/tools/{canonical_name}.j2"
        try:
            template = self.template_service.get_template(template_path)
            rendered = template.render(
                namespace=tool_namespace, few_shots=str(few_shots)
            )
            return PromptTemplate(template=rendered.strip())
        except TemplateNotFound:
            return PromptTemplate(template="")
        except Exception as exc:
            logger.warning("PromptBuilder: failed to render %s: %s", template_path, exc)
            return PromptTemplate(template="")

    def create_citation_guidelines(
        self,
        generate_citations: bool = True,
        documents: list[Document] | None = None,
        nodes: list[NodeWithScore] | None = None,
        few_shots: bool = True,
    ) -> BasePromptTemplate:
        if not generate_citations:
            return PromptTemplate(template="")

        if documents is None and nodes is not None:
            documents = [Document.from_node(node) for node in nodes]

        if not documents:
            return PromptTemplate(template="")

        sources = [format_llm_source(doc) for doc in documents]
        return self.template_service.create_prompt_template(
            "chat/guidelines/citations.j2",
            all_cites=", ".join(sources),
            sample_cite_1=sources[0],
            sample_cite_2=sources[min(len(sources) - 1, 1)],
            few_shots=few_shots,
        )

    def create_thinking_guidelines(self, few_shots: bool = True) -> BasePromptTemplate:
        """Create the thinking/reasoning guidelines prompt.

        Returns an empty ``PromptTemplate`` when the template is not found or
        rendering fails.
        """
        return self.template_service.create_prompt_template(
            "chat/guidelines/thinking.j2",
            few_shots=few_shots,
        )

    def seed_tool_instructions(self, tools: list["ToolSpec"]) -> list["ToolSpec"]:
        """Seed tool instructions from Jinja templates for tools that lack them.

        Tools with explicit ``instructions`` (including empty string to suppress)
        are returned unchanged. The prompt builder owns all template-rendering
        responsibility; callers just pass the tool list in and get it back.
        """
        namespace = _build_tool_namespace(tools)
        return [self._seed_one_tool(tool, namespace) for tool in tools]

    def _seed_one_tool(
        self, tool: "ToolSpec", namespace: "_ToolNamespace"
    ) -> "ToolSpec":
        if tool.instructions is not None:
            return tool
        try:
            canonical = tool.get_original_tool_name()
        except ValueError:
            canonical = tool.name or ""
        rendered = self.create_tool_instructions(canonical, namespace).format().strip()
        if rendered:
            return tool.model_copy(update={"instructions": rendered})
        return tool
