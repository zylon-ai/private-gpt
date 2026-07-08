"""Shared eager-loading of injector-bound components.

Used by both the FastAPI server (``launcher.py``) and the long-lived Celery
chat worker so that both processes warm the exact same dependency graph.
Keeping this in one place guarantees the chat worker mirrors the server's DI
state instead of drifting.
"""
import logging

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


def eager_loading(injector: Injector) -> None:
    """Eagerly load modules to avoid race conditions in multi-threaded environments."""
    logger.debug("Initializing mandatory dependencies")
    injector.get(Settings)

    # Models
    logger.debug("Initializing models")
    injector.get(LLMComponent)
    injector.get(EmbeddingComponent)

    # Stores
    logger.debug("Initializing stores")
    injector.get(NodeStoreComponent)
    injector.get(VectorStoreComponent)

    # Streaming components
    logger.debug("Initializing streaming components")
    injector.get(StreamComponent)
    injector.get(StreamManager)

    # Auxiliar
    logger.debug("Initializing auxiliar services")
    injector.get(PromptBuilderService)
    injector.get(ToolService)
    injector.get(CodeExecutionComponent)


def warm_chat_components(injector: Injector) -> None:
    """Resolve chat-path singletons so the first chat task is not cold.

    Called by the chat worker on startup (after ``eager_loading``) to make the
    first request pay no warm-up cost. ``ChatService`` and friends are
    singletons, so this resolves them once and they are reused for the whole
    worker lifetime.
    """
    from private_gpt.components.streaming.stream.stream_processor import (
        StreamProcessor,
    )
    from private_gpt.server.chat.chat_service import ChatService

    logger.debug("Warming chat-path components")
    injector.get(ChatService)
    injector.get(StreamProcessor)
