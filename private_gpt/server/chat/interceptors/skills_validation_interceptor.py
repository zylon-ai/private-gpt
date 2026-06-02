from collections.abc import Sequence

from injector import inject, singleton

from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
    SkillsRuntimeCache,
)
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.server.utils.artifact_input import ArtifactType, SkillArtifact


@singleton
class SkillsValidationInterceptor(ChatRequestLoopInterceptor):
    """Resolve and validate skills once, then cache results in runtime state."""

    @inject
    def __init__(self, skill_service: SkillService) -> None:
        self._skill_service = skill_service

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase not in {
            InterceptorPhase.VALIDATION,
            InterceptorPhase.BEFORE_ITERATION,
        }:
            return

        state = context.state
        skill_filter = self._find_skill_filter(state.input.request.tool_context)
        if skill_filter is None:
            return

        if state.runtime.cache.skill is not None:
            # Cache is already prepared for this run.
            context.set_state(state)
            return

        entries = await self._skill_service.recover_versions(skill_filter)
        state.runtime.cache.skill = SkillsRuntimeCache(
            entries=entries,
        )
        context.set_state(state)

    def _find_skill_filter(
        self, tool_context: Sequence[ArtifactType]
    ) -> SkillFilter | None:
        for artifact in tool_context:
            if isinstance(artifact, SkillArtifact):
                return artifact.skill_filter
        return None
