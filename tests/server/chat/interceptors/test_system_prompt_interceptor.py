"""Tests for SystemPromptRequestInterceptor layer deduplication.

Verifies that running the interceptor N times (simulating tool-call loops or
the recalculate branch) never accumulates duplicate layers in the context
stack or duplicates text in the rendered system prompt.
"""

from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
)
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.components.engines.chat.utils.request_builder import (
    build_initial_context_stack,
)
from private_gpt.server.chat.interceptors.system_prompt_interceptor import (
    SystemPromptRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

_SYSTEM_PROMPT = "You are Zylon, an AI assistant.\nCurrent date: 2026-07-27."


def _make_request(
    system_prompt: str | list[TextBlock] | None = _SYSTEM_PROMPT,
) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt=system_prompt),
    )


def _make_context(
    request: ResolvedChatRequest,
    context_stack: ContextStack | None = None,
    phase: InterceptorPhase = InterceptorPhase.BEFORE_ITERATION,
) -> ChatInterceptorContext:
    stack = (
        context_stack
        if context_stack is not None
        else build_initial_context_stack(request)
    )
    state = ChatState(
        input=ChatInputState(
            request=request,
            context_stack=stack,
        ),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
    )
    return ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=phase,
        emit_fn=lambda _: None,
    )


def _make_interceptor(
    add_context_to_system_prompt: bool = False,
) -> SystemPromptRequestInterceptor:
    """Build a SystemPromptRequestInterceptor with a minimal PromptBuilderService."""
    prompt_builder = MagicMock()
    prompt_template = MagicMock()
    prompt_template.format.return_value = _SYSTEM_PROMPT
    prompt_builder.create_chat_header_prompt.return_value = prompt_template
    return SystemPromptRequestInterceptor(
        prompt_builder_service=prompt_builder,
        add_context_to_system_prompt=add_context_to_system_prompt,
    )


class TestSystemPromptInterceptorIdempotency:
    """The interceptor must be idempotent across repeated calls."""

    @pytest.mark.asyncio
    async def test_single_run_produces_one_platform_header_layer(self) -> None:
        interceptor = _make_interceptor()
        request = _make_request()
        context = _make_context(request)

        await interceptor.intercept(context)

        platform_layers = [
            layer
            for layer in context.state.input.context_stack.layers
            if layer.source == "platform_header"
        ]
        assert len(platform_layers) == 1, "Expected exactly 1 platform_header layer"

    @pytest.mark.asyncio
    async def test_running_n_times_does_not_accumulate_layers(self) -> None:
        """Simulates multiple BEFORE_ITERATION passes (tool-call loop)."""
        interceptor = _make_interceptor()
        request = _make_request()
        context = _make_context(request)

        for _ in range(5):
            await interceptor.intercept(context)

        platform_layers = [
            layer
            for layer in context.state.input.context_stack.layers
            if layer.source == "platform_header"
        ]
        assert len(platform_layers) == 1, (
            f"After 5 iterations got {len(platform_layers)} platform_header layers — "
            "interceptor is accumulating duplicates!"
        )

    @pytest.mark.asyncio
    async def test_running_n_times_no_user_instruction_duplication(self) -> None:
        """Multiple BEFORE_ITERATION passes: user instructions appear once."""
        interceptor = _make_interceptor()
        request = _make_request()
        context = _make_context(request)

        for _ in range(5):
            await interceptor.intercept(context)

        user_layers = [
            layer
            for layer in context.state.input.context_stack.layers
            if layer.type == LayerType.USER_INSTRUCTIONS
        ]
        sources = [layer.source for layer in user_layers]
        assert sources.count("request") <= 1, (
            f"'request' USER_INSTRUCTIONS layer duplicated: {sources}"
        )
        assert sources.count("platform_header") <= 1, (
            f"'platform_header' USER_INSTRUCTIONS layer duplicated: {sources}"
        )

    @pytest.mark.asyncio
    async def test_system_prompt_text_not_duplicated_after_n_iterations(self) -> None:
        """The rendered system prompt text must not repeat after N runs."""
        interceptor = _make_interceptor()
        request = _make_request()
        context = _make_context(request)

        for _ in range(5):
            await interceptor.intercept(context)

        prompt = context.state.input.request.system.prompt
        # Normalise to list of text strings
        if isinstance(prompt, str):
            texts = [prompt]
        elif isinstance(prompt, list):
            texts = [b.text for b in prompt if isinstance(b, TextBlock) and b.text]
        else:
            texts = []

        full_text = "\n".join(texts)
        occurrences = full_text.count(_SYSTEM_PROMPT)
        assert (
            occurrences <= 2
        ), (  # at most 2: once in user layer, once in platform_header
            f"System prompt text appears {occurrences} times after 5 iterations. "
            "Likely a duplication bug!"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_saves_original_not_mutated_prompt(self) -> None:
        """Verify checkpoint round-trip with original request avoids duplication.

        The ResumableChatRunner now saves ``state.original_input.request``.
        This test simulates the resume path where the original (clean) request
        is used together with ``build_initial_context_stack``.
        """
        interceptor = _make_interceptor()
        original_request = _make_request()

        # --- Simulate first request execution ---
        context = _make_context(original_request)
        await interceptor.intercept(context)

        # Verify _render_system_prompt_text returns a single TextBlock
        mutated_prompt = context.state.input.request.system.prompt
        if isinstance(mutated_prompt, list):
            assert len(mutated_prompt) == 1, (
                "_render_system_prompt_text should return exactly 1 TextBlock"
            )

        # --- Simulate resume: build fresh stack from ORIGINAL (clean) request ---
        restored_stack = build_initial_context_stack(original_request)
        restored_context = _make_context(original_request, context_stack=restored_stack)
        await interceptor.intercept(restored_context)

        prompt = restored_context.state.input.request.system.prompt
        if isinstance(prompt, str):
            full_text = prompt
        elif isinstance(prompt, list):
            full_text = "\n".join(
                b.text for b in prompt if isinstance(b, TextBlock) and b.text
            )
        else:
            full_text = ""

        occurrences = full_text.count(_SYSTEM_PROMPT)
        assert occurrences <= 2, (
            f"After resume the system prompt appears {occurrences} times."
        )

    @pytest.mark.asyncio
    async def test_round_tripped_prompt_does_not_duplicate_rendered_blocks(
        self,
    ) -> None:
        """A request whose ``system.prompt`` already carries the rendered
        stack (client echoes back a previous response) must not produce a
        system message whose content blocks repeat the header or guidelines
        that the interceptors regenerate as isolated layers.

        Layers stay isolated in the stack; at render time duplicate rendered
        text is discarded so the snowball never reaches the LLM.
        """
        header = "You are Zylon, an AI assistant.\nCurrent date: 2026-07-28."
        guideline = "<response_formatting>\nWrite clearly.\n</response_formatting>"
        # Client re-sends the fully rendered prompt (header + guideline baked in)
        bloated_prompt = f"{header}\n\n{guideline}"

        interceptor = _make_interceptor()
        # Header template renders the same header the client already embedded
        interceptor._prompt_builder_service.create_chat_header_prompt.return_value.format.return_value = (
            header
        )
        request = _make_request(system_prompt=bloated_prompt)
        context = _make_context(request)
        await interceptor.intercept(context)

        prompt = context.state.input.request.system.prompt
        if isinstance(prompt, str):
            texts = [prompt]
        elif isinstance(prompt, list):
            texts = [b.text for b in prompt if isinstance(b, TextBlock) and b.text]
        else:
            texts = []

        full = "\n".join(texts)
        # Header must appear at most once across all rendered blocks
        assert full.count(header) <= 1, (
            f"Header rendered {full.count(header)} times after round-trip: {texts!r}"
        )
        # Guideline must appear at most once
        assert full.count(guideline) <= 1, (
            f"Guideline rendered {full.count(guideline)} times: {texts!r}"
        )
        # No rendered block may fully duplicate another kept block
        stripped = [t.strip() for t in texts if t.strip()]
        for i, block in enumerate(stripped):
            for j, other in enumerate(stripped):
                if i != j and block and block in other:
                    raise AssertionError(
                        f"Rendered block #{i} is contained in block #{j} — "
                        "duplicate content reached the rendered prompt."
                    )

    @pytest.mark.asyncio
    async def test_fallback_build_with_mutated_prompt_is_safe(self) -> None:
        """Defensive: even if build_initial is called on a mutated request,
        the system prompt should not explode (snowball test)."""
        interceptor = _make_interceptor()
        request = _make_request()
        context = _make_context(request)
        await interceptor.intercept(context)

        # Simulate mutated request being re-ingested
        mutated_request = context.state.input.request
        restored_stack = build_initial_context_stack(mutated_request)
        restored_context = _make_context(mutated_request, context_stack=restored_stack)
        await interceptor.intercept(restored_context)

        prompt = restored_context.state.input.request.system.prompt
        if isinstance(prompt, str):
            full_text = prompt
        elif isinstance(prompt, list):
            full_text = "\n".join(
                b.text for b in prompt if isinstance(b, TextBlock) and b.text
            )
        else:
            full_text = ""

        occurrences = full_text.count(_SYSTEM_PROMPT)
        assert occurrences <= 3, (
            f"Even on fallback path, the system prompt explodes to {occurrences} occurrences."
        )
