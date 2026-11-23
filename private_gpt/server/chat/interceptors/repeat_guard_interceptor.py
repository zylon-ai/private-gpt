import logging

from injector import inject, singleton

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.utils.loop_guard import (
    LoopGuardDirective,
    LoopGuardLevel,
    analyze_repeats,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class RepeatGuardRequestInterceptor(ChatRequestLoopInterceptor):
    """Escalate how the history is shown to the LLM when tool steps repeat.

    Counts identical tool calls and identical tool results since the latest user
    message (not iterations). The level is derived from the history on every
    iteration, so it survives resumes without extra state:

    * any repeated step                                   -> NOTE
    * the latest call/result seen ``flatten_after`` times -> FLATTEN
    * ``final_after`` repeated steps in total             -> FINAL (no tools)
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def intercept(self, context: ChatInterceptorContext) -> None:
        guard = self._settings.chat.repeat_guard
        if not guard.enabled or context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        analysis = analyze_repeats(context.state.input.request.messages)
        repeated_total = len(analysis.repeated_indexes)
        if not repeated_total:
            return

        latest = analysis.latest_repeat_count
        if repeated_total >= guard.final_after:
            level = LoopGuardLevel.FINAL
        elif latest >= guard.flatten_after:
            level = LoopGuardLevel.FLATTEN
        else:
            level = LoopGuardLevel.NOTE

        reason = (
            f"latest step seen {latest} times; "
            f"{repeated_total} repeated steps since the latest user message"
        )
        if level >= LoopGuardLevel.FLATTEN:
            logger.info("Repeat guard escalated to %s: %s", level.name, reason)
        context.state.input.loop_guard = LoopGuardDirective(level=level, reason=reason)
        context.set_state(context.state)
