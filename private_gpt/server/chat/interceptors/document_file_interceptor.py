import logging

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    preprocess_document_history,
)
from private_gpt.components.chat.processors.chat_history.documents.document_uploader import (
    upload_document_history,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.ingestion.ingestion_scheduler import (
    IngestionSchedulerFactory,
)
from private_gpt.components.tools.processors.base import (
    _is_unresolved_tool,
    _tool_matches,
    session_id_for,
)
from private_gpt.components.tools.tool_names import (
    CODE_EXECUTION_INTERNAL_TOOLS,
    CONVERT_DOCUMENTS_TOOL_NAME,
)
from private_gpt.server.chat.interceptors.preprocessing_tool_calls import (
    PreprocessingToolCalls,
)
from private_gpt.server.files.file_service import FileService
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

DOCUMENT_PROCESSING_TOOL_NAME = "document_preprocessing"


@singleton
class DocumentFilePreprocessingInterceptor(ChatRequestLoopInterceptor):
    """Make attached documents usable, either as text or as sandbox files."""

    @inject
    def __init__(
        self,
        scheduler_factory: IngestionSchedulerFactory,
        file_service: FileService,
        settings: Settings,
    ) -> None:
        self._scheduler_factory = scheduler_factory
        self._file_service = file_service
        self._tool_name = DOCUMENT_PROCESSING_TOOL_NAME
        self._preprocess_settings = settings.chat.preprocess.documents

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Convert document blocks to text, or land them in the sandbox.

        This runs only on the first iteration of a request. Once a document
        block has been converted to plain text it is not present in later
        iterations; later iterations still see the updated message history
        through the document/citation/system-prompt interceptors, which are
        intentionally *not* gated.

        When the request carries code execution the documents are written to the
        session uploads mount instead of being inlined, and the model reaches
        their content through the ``convert_documents`` tool.
        """
        if (
            context.phase != InterceptorPhase.BEFORE_ITERATION
            or context.state.runtime.iteration > 0
        ):
            return

        if await self._upload_to_sandbox(context):
            return

        await self._inline_as_text(context)

    def _uses_sandbox_documents(self, tools: list[ToolSpec]) -> bool:
        """Return True when documents should land in the sandbox as files.

        Requires both a code execution tool *and* a resolved
        ``convert_documents``: without the latter the bytes would sit on disk
        with no way for the model to read their content, so we would rather
        fall back to inlining than degrade silently.
        """
        has_code_execution = any(
            _tool_matches(tool, *CODE_EXECUTION_INTERNAL_TOOLS) for tool in tools
        )
        has_converter = any(
            _tool_matches(tool, CONVERT_DOCUMENTS_TOOL_NAME)
            and not _is_unresolved_tool(tool)
            for tool in tools
        )
        return has_code_execution and has_converter

    async def _upload_to_sandbox(self, context: ChatInterceptorContext) -> bool:
        """Persist attachments as sandbox files. Returns False to fall back."""
        state = context.state
        tools = (
            state.input.context_stack.all_tools()
            or state.input.request.tool_config.tools
        )
        if not self._uses_sandbox_documents(tools):
            return False

        try:
            messages = await upload_document_history(
                chat_history=state.input.request.messages,
                file_service=self._file_service,
                scope_id=session_id_for(state.input.request),
                convert_service=self._scheduler_factory.get(),
                max_concurrency=self._preprocess_settings.max_concurrency,
            )
        except Exception:
            logger.exception(
                "Could not upload attachments to the sandbox; "
                "falling back to inline conversion"
            )
            return False

        if messages is None:
            logger.warning(
                "Every attachment upload failed; falling back to inline conversion"
            )
            return False

        state.input.request.messages = messages
        context.set_state(state)
        return True

    async def _inline_as_text(self, context: ChatInterceptorContext) -> None:
        """Convert attachments to markdown and inject them into the message."""
        state = context.state
        tool_calls = PreprocessingToolCalls(
            context,
            tool_name=self._tool_name,
            return_type=self._preprocess_settings.return_type,
            default_error="There was an error during document processing.",
        )

        try:
            async for response in preprocess_document_history(
                chat_history=state.input.request.messages,
                convert_service=self._scheduler_factory.get(),
                max_concurrency=self._preprocess_settings.max_concurrency,
                return_type=self._preprocess_settings.return_type,
            ):
                processing = response.processing_status
                if processing is not None:
                    if processing.status == "processing":
                        tool_calls.start(
                            processing.doc_index,
                            {
                                "type": "document",
                                "index": processing.doc_index,
                                "name": processing.reference,
                            },
                        )
                    elif processing.status in {"completed", "failed"}:
                        tool_calls.finish(
                            processing.doc_index,
                            processing.content,
                            is_error=processing.status == "failed",
                            error_detail=processing.error_detail,
                        )

                if response.chat_history is not None:
                    state.input.request.messages = response.chat_history
        except Exception as exc:
            if not tool_calls.has_pending:
                raise
            logger.exception("Document preprocessing failed; reporting as tool error")
            tool_calls.fail_pending(exc)

        state.input.request.messages = tool_calls.append_tool_messages(
            state.input.request.messages
        )
        context.set_state(state)
