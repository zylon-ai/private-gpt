"""Profile-based eager-loading of injector-bound components.

A single entry point ``warm(injector, profile)`` dispatches to the right
component groups based on the role. New worker types only need a new
profile entry — no changes anywhere else.
"""
import logging
from collections.abc import Callable, Sequence

from injector import Injector

from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.streaming.stream.stream_manager import StreamManager
from private_gpt.components.streaming.stream_component import StreamComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.tools.tool_service import ToolService
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_Warmer = Callable[[Injector], None]


def _warm_base(injector: Injector) -> None:
    logger.debug("Warming base (settings + models)")
    injector.get(Settings)
    injector.get(LLMComponent)
    injector.get(EmbeddingComponent)


def _warm_stores(injector: Injector) -> None:
    logger.debug("Warming stores")
    injector.get(NodeStoreComponent)
    injector.get(VectorStoreComponent)


def _warm_streaming(injector: Injector) -> None:
    logger.debug("Warming streaming components")
    injector.get(StreamComponent)
    injector.get(StreamManager)


def _warm_tools(injector: Injector) -> None:
    logger.debug("Warming tools")
    injector.get(PromptBuilderService)
    injector.get(ToolService)
    injector.get(CodeExecutionComponent)


def _warm_chat(injector: Injector) -> None:
    from private_gpt.components.streaming.stream.stream_processor import (
        StreamProcessor,
    )
    from private_gpt.server.chat.chat_service import ChatService

    logger.debug("Warming chat-path components")
    injector.get(ChatService)
    injector.get(StreamProcessor)


_GROUPS: dict[str, _Warmer] = {
    "base": _warm_base,
    "stores": _warm_stores,
    "streaming": _warm_streaming,
    "tools": _warm_tools,
    "chat": _warm_chat,
}

_PROFILES: dict[str, Sequence[str]] = {
    "chat": ("base", "stores", "streaming", "tools", "chat"),
    "tools": ("base", "tools"),
    "full": ("base", "stores", "streaming", "tools"),
}


def warm(injector: Injector, profile: str = "full") -> None:
    """Eager-resolve all DI singletons for the given worker profile.

    ``profile`` is one of ``chat``, ``tools``, or ``full`` (API server).
    """
    groups = _PROFILES.get(profile)
    if groups is None:
        raise ValueError(
            f"Unknown warm-up profile {profile!r}. "
            f"Available: {', '.join(sorted(_PROFILES))}"
        )
    logger.info("Warming profile=%s (groups: %s)", profile, ", ".join(groups))
    for group in groups:
        _GROUPS[group](injector)


def eager_loading(injector: Injector) -> None:
    """Full warm-up: everything. Kept for API server backward compat."""
    warm(injector, "full")
