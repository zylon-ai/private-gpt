import logging
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import (
    ToolInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.tool_names import (
    CODE_EXECUTION_INTERNAL_TOOLS,
    SKILL_MANAGEMENT_TOOLS,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.chat.input_models import PromptConfig


logger = logging.getLogger(__name__)

_SOURCE_TOOL_INSTRUCTIONS = "platform:tool_instructions"
_SOURCE_CITATIONS = "platform:citations"
_SOURCE_THINKING = "platform:thinking"
_SOURCE_CODE_EXECUTION = "platform:code_execution"
_SOURCE_SKILLS = "platform:skills"

_ALL_PLATFORM_SOURCES = frozenset(
    {
        _SOURCE_TOOL_INSTRUCTIONS,
        _SOURCE_CITATIONS,
        _SOURCE_THINKING,
        _SOURCE_CODE_EXECUTION,
        _SOURCE_SKILLS,
    }
)


@singleton
class PlatformGuidelinesInterceptor(ChatRequestLoopInterceptor):
    """Injects all platform-level prompt layers before each LLM iteration."""

    @inject
    def __init__(
        self, prompt_builder: PromptBuilderService, settings: Settings
    ) -> None:
        self._prompt_builder = prompt_builder
        self._skill_injection_mode = settings.skills.skill_injection_mode

    async def intercept(self, context: ChatInterceptorContext) -> None:
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
            stack = self._inject_tool_instructions(
                stack, tools, _SOURCE_TOOL_INSTRUCTIONS
            )

        # 2. Citation formatting guidelines
        documents = stack.all_documents()
        if prompt.citations and documents:
            stack = self._inject_citations(stack, _SOURCE_CITATIONS)

        # 3. Thinking guidelines (only when reasoning is enabled)
        if prompt.thinking and request.thinking.enabled:
            stack = self._inject_thinking(stack, _SOURCE_THINKING)

        # 4. Code execution environment instructions
        if prompt.code_execution and self._has_code_execution_tool(tools):
            stack = self._inject_code_execution(stack, tools, _SOURCE_CODE_EXECUTION)

        # 5. Skill management instructions
        if prompt.skills and self._has_skill_management_tool(tools):
            stack = self._inject_skills(stack, tools, _SOURCE_SKILLS)

        state.input.context_stack = stack
        context.set_state(state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_tool_instructions(
        self, stack: ContextStack, tools: list[ToolSpec], source: str
    ) -> ContextStack:
        # Remove any old tool-instruction layers from prior iterations
        # to prevent prompt accumulation across chain repeats.
        stack = stack.remove_layers_of_source(source)

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
                    source=source,
                )
            )

        return stack

    def _inject_citations(self, stack: ContextStack, source: str) -> ContextStack:
        # Remove any old citation layer from prior iterations
        # to prevent prompt accumulation across chain repeats.
        stack = stack.remove_layers_of_source(source)

        documents = stack.all_documents()
        content = self._get_citation_guidelines_content(documents=documents)
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="citations",
                    instructions=content,
                    source=source,
                )
            )
        return stack

    def _inject_thinking(self, stack: ContextStack, source: str) -> ContextStack:
        # Remove any old thinking layer from prior iterations
        # to prevent prompt accumulation across chain repeats.
        stack = stack.remove_layers_of_source(source)

        content = self._get_thinking_content()
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="thinking",
                    instructions=content,
                    source=source,
                )
            )
        return stack

    def _inject_code_execution(
        self, stack: ContextStack, tools: list[ToolSpec], source: str
    ) -> ContextStack:
        # Remove any old code-execution layer from prior iterations
        # to prevent prompt accumulation across chain repeats.
        stack = stack.remove_layers_of_source(source)

        content = self._prompt_builder.create_code_execution_prompt(tools).format()
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="bash",
                    instructions=content,
                    source=source,
                )
            )
        return stack

    def _inject_skills(
        self, stack: ContextStack, tools: list[ToolSpec], source: str
    ) -> ContextStack:
        # Remove any old skill-instructions layer from prior iterations
        # to prevent prompt accumulation across chain repeats.
        stack = stack.remove_layers_of_source(source)

        content = self._prompt_builder.create_skills_prompt(tools).format()
        if content:
            stack = stack.append_layer(
                ToolInstructionsLayer(
                    tool_name="skills",
                    instructions=content,
                    source=source,
                )
            )
        return stack

    @staticmethod
    def _has_code_execution_tool(tools: list[ToolSpec]) -> bool:
        for tool in tools:
            try:
                canonical = tool.get_original_tool_name()
            except ValueError:
                canonical = tool.name or ""
            if canonical in CODE_EXECUTION_INTERNAL_TOOLS:
                return True
        return False

    @staticmethod
    def _has_skill_management_tool(tools: list[ToolSpec]) -> bool:
        for tool in tools:
            try:
                canonical = tool.get_original_tool_name()
            except ValueError:
                canonical = tool.name or ""
            if canonical in SKILL_MANAGEMENT_TOOLS:
                return True
        return False

    # ------------------------------------------------------------------
    # Cached content loaders
    # ------------------------------------------------------------------

    def _get_thinking_content(self) -> str:
        return self._prompt_builder.create_thinking_guidelines().format()

    def _get_citation_guidelines_content(
        self, documents: list[Document] | None = None
    ) -> str:
        return self._prompt_builder.create_citation_guidelines(
            documents=documents
        ).format()
