import logging

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.chat.processors.chat_history.multimodality.media_uploader import (
    upload_media_history,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.tools.processors.base import (
    _is_unresolved_tool,
    _tool_matches,
    session_id_for,
)
from private_gpt.components.tools.tool_names import (
    CODE_EXECUTION_INTERNAL_TOOLS,
    DESCRIBE_IMAGE_TOOL_NAME,
    TRANSCRIBE_AUDIO_TOOL_NAME,
)
from private_gpt.server.files.file_service import FileService
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class MediaFilePreprocessingInterceptor(ChatRequestLoopInterceptor):
    """Also land attached images and audio in the sandbox, as files.

    This runs before the multimodal interceptor and is deliberately additive: it
    appends a note and nothing else. Whatever happens next — the blocks going to
    the model natively, or being replaced by a description — is unchanged.

    Landing the bytes on disk is what lets a multimodal model *compute* over an
    attachment rather than only look at it, and it rescues the media that would
    otherwise be silently dropped: everything the multimodal budget trims from
    older messages, and anything attached to a non-user message.
    """

    @inject
    def __init__(self, file_service: FileService, settings: Settings) -> None:
        self._file_service = file_service
        self._preprocess_settings = settings.chat.preprocess.multimodal

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if (
            context.phase != InterceptorPhase.BEFORE_ITERATION
            or context.state.runtime.iteration > 0
        ):
            return

        state = context.state
        tools = (
            state.input.context_stack.all_tools()
            or state.input.request.tool_config.tools
        )
        if not self._uses_sandbox_media(tools):
            return

        try:
            messages = await upload_media_history(
                chat_history=state.input.request.messages,
                file_service=self._file_service,
                scope_id=session_id_for(state.input.request),
                max_concurrency=self._preprocess_settings.max_concurrency,
            )
        except Exception:
            # Nothing was taken away, so a failure here costs the sandbox copy
            # and nothing else. The request continues exactly as it would have.
            logger.exception("Could not save media attachments to the sandbox")
            return

        if messages is None:
            return

        state.input.request.messages = messages
        context.set_state(state)

    def _uses_sandbox_media(self, tools: list[ToolSpec]) -> bool:
        """Return True when media should also land in the sandbox as files.

        Mirrors the document gate: a code execution tool *and* at least one
        resolved media tool. Either tool is enough on its own — with code
        execution present the model can read the bytes whether or not it can
        also ask for a description or a transcript.
        """
        has_code_execution = any(
            _tool_matches(tool, *CODE_EXECUTION_INTERNAL_TOOLS) for tool in tools
        )
        has_media_tool = any(
            _tool_matches(tool, DESCRIBE_IMAGE_TOOL_NAME, TRANSCRIBE_AUDIO_TOOL_NAME)
            and not _is_unresolved_tool(tool)
            for tool in tools
        )
        return has_code_execution and has_media_tool
