import logging
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import (
    ToolInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.prompts.prompt_builder import PromptBuilderService

if TYPE_CHECKING:
    from private_gpt.chat.input_models import PromptConfig


logger = logging.getLogger(__name__)

_SOURCE_TOOL_INSTRUCTIONS = "platform:tool_instructions"
_SOURCE_CITATIONS = "platform:citations"
_SOURCE_THINKING = "platform:thinking"

_ALL_PLATFORM_SOURCES = frozenset(
    {_SOURCE_TOOL_INSTRUCTIONS, _SOURCE_CITATIONS, _SOURCE_THINKING}
)


@singleton
class PlatformGuidelinesInterceptor(ChatRequestLoopInterceptor):
    """Injects all platform-level prompt layers before each LLM iteration."""

    @inject
    def __init__(self, prompt_builder: PromptBuilderService) -> None:
        self._prompt_builder = prompt_builder
        self._thinking_content: str | None = None
        self._citation_guidelines_content: str | None = None

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        request = state.input.request
        prompt: PromptConfig = request.system.platform_prompts

        # Remove all previously injected platform layers.
        stack = state.input.context_stack
        for source in _ALL_PLATFORM_SOURCES:
            stack = stack.remove_layers_of_source(source)

        tools = stack.all_tools()

        # 1. Per-tool instructions (ToolSpec.instructions).
        #    Seeded by InternalToolRequestInterceptor for internal tools, or provided
        #    explicitly by the caller for external tools.
        if prompt.tools:
            stack = self._inject_tool_instructions(stack, tools)

        # 2. Citation formatting guidelines
        documents = stack.all_documents()
        if prompt.citations and documents:
            stack = self._inject_citations(stack)

        # 3. Thinking guidelines (only when reasoning is enabled)
        if prompt.thinking and request.thinking.enabled:
            stack = self._inject_thinking(stack)

        state.input.context_stack = stack
        context.set_state(state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_tool_instructions(
        self, stack: ContextStack, tools: list[ToolSpec]
    ) -> ContextStack:
        for tool in tools:
            if tool.name is None or not tool.instructions:
                continue

            try:
                canonical = tool.get_original_tool_name()
            except ValueError:
                canonical = tool.name

            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name=canonical,
                    instructions=tool.instructions,
                    source=_SOURCE_TOOL_INSTRUCTIONS,
                )
            )

        return stack

    def _inject_citations(self, stack: ContextStack) -> ContextStack:
        documents = stack.all_documents()
        content = self._get_citation_guidelines_content(documents=documents)
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="citations",
                    instructions=content,
                    source=_SOURCE_CITATIONS,
                )
            )
        return stack

    def _inject_thinking(self, stack: ContextStack) -> ContextStack:
        content = self._get_thinking_content()
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="thinking",
                    instructions=content,
                    source=_SOURCE_THINKING,
                )
            )
        return stack

    # ------------------------------------------------------------------
    # Cached content loaders
    # ------------------------------------------------------------------

    def _get_thinking_content(self) -> str:
        if self._thinking_content is None:
            self._thinking_content = (
                self._prompt_builder.create_thinking_guidelines().format()
            )
        return self._thinking_content

    def _get_citation_guidelines_content(
        self, documents: list[Document] | None = None
    ) -> str:
        if self._citation_guidelines_content is None:
            self._citation_guidelines_content = (
                self._prompt_builder.create_citation_guidelines(
                    documents=documents
                ).format()
            )
        return self._citation_guidelines_content
