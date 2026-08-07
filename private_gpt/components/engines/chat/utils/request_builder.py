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
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.sandbox.mount import Mount


def build_initial_context_stack(
    request: ChatRequest, source: str = "request"
) -> ContextStack:
    """Create the initial context stack from user-provided request data."""
    stack = ContextStack()

    if isinstance(request, ResolvedChatRequest):
        # We only include system prompt, tools, and documents
        # in the context stack if they are present in the request.

        if request.system.prompt:
            stack = stack.remove_layers_of_type(LayerType.USER_INSTRUCTIONS)
            stack = stack.append_layer(
                UserInstructionsLayer(text=request.system.prompt, source=source)
            )

        if request.tool_config.tools:
            stack = stack.remove_layers_of_type(LayerType.TOOL_DEFINITIONS)
            stack = stack.append_layer(
                ToolDefinitionsLayer(
                    tools=list(request.tool_config.tools),
                    source=source,
                )
            )

        if request.context.documents:
            stack = stack.remove_layers_of_type(LayerType.DOCUMENT)
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
    # The initial ResolvedChatRequest already carries Backend mount-plan
    # volumes (resolved from ChatBody.mounts by ChatRequestMapper). Keep them
    # and append skill/bundle mounts from the context stack — the single
    # mount set that tools later pass to the environment manager. The merge
    # is idempotent: build_request_from_context_stack runs on every LLM
    # iteration, so duplicate mounts would otherwise accumulate.
    request.context.mounts = _merge_mounts(
        request.context.mounts, context_stack.all_mounts()
    )

    request.messages = [m for m in request.messages if m.role != MessageRole.SYSTEM]
    request.system.prompt = _render_system_prompt_text(context_stack)

    return request


def _merge_mounts(*groups: list[Mount]) -> list[Mount]:
    """Merge mount groups, deduplicating by mount identity.

    ``Mount`` carries a ``storage`` ref with an excluded callable, so the
    identity is the target + access + source + storage prefix — enough to keep
    skills and mount-plan volumes stable across repeated request builds.
    """
    seen: set[tuple[object, ...]] = set()
    merged: list[Mount] = []
    for group in groups:
        for mount in group:
            key = (
                mount.target,
                mount.access,
                str(mount.source) if mount.source is not None else "",
                mount.storage.prefix if mount.storage is not None else "",
            )
            if key not in seen:
                seen.add(key)
                merged.append(mount)
    return merged


def _render_system_prompt_text(context_stack: ContextStack) -> list[TextBlock] | None:
    """Join prompt layers into a single system prompt string."""
    blocks = context_stack.to_system_prompt()
    if not blocks:
        return None

    parts = [block.text for block in blocks if block.text and block.text.strip()]
    if not parts:
        return None

    return [TextBlock(text=part) for part in parts]
