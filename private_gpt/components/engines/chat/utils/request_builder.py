from llama_index.core.base.llms.types import MessageRole, TextBlock

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
)
from private_gpt.components.context.models.context_layer import (
    DocumentLayer,
    ToolDefinitionsLayer,
    UserInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack


def build_initial_context_stack(
    request: ChatRequest, source: str = "request"
) -> ContextStack:
    """Create the initial context stack from user-provided request data."""
    stack = ContextStack()

    if isinstance(request, ResolvedChatRequest):
        # We only include system prompt, tools, and documents
        # in the context stack if they are present in the request.

        if request.system.prompt:
            stack = stack.append_layer(
                UserInstructionsLayer(text=request.system.prompt, source=source)
            )

        if request.tool_config.tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(
                    tools=list(request.tool_config.tools),
                    source=source,
                )
            )

        if request.context.documents:
            for document in request.context.documents:
                stack = stack.append_layer(
                    DocumentLayer(document=document, source=source)
                )

    return stack


def build_request_from_context_stack(
    base_request: ChatRequest,
    context_stack: ContextStack,
) -> ResolvedChatRequest:
    """Materialize a ChatRequest from the latest context stack layers."""
    request = ResolvedChatRequest.model_validate(base_request, from_attributes=True)

    request.tool_config.tools = list(context_stack.all_tools())
    request.context.documents = context_stack.all_documents() or None
    request.context.content_bundles = context_stack.all_bundles()
    request.context.bundles_to_remove = context_stack.all_bundles_to_remove()

    request.messages = [m for m in request.messages if m.role != MessageRole.SYSTEM]
    request.system.prompt = _render_system_prompt_text(context_stack)

    return request


def _render_system_prompt_text(context_stack: ContextStack) -> list[TextBlock] | None:
    """Join prompt layers into a single system prompt string."""
    blocks = context_stack.to_system_prompt()
    if not blocks:
        return None

    parts = [block.text for block in blocks if block.text and block.text.strip()]
    if not parts:
        return None

    return [TextBlock(text=part) for part in parts]
