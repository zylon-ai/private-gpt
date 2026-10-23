"""Request-scoped context captured at tool-execution time.

Server tools are built once at VALIDATION, before request-scoped state such as
loaded-skill mounts exists. When the tool executes, the current request state
is available and must be reflected in the tool's session config so the sandbox
is created with the right volumes / env.

This is a *typed, generic* bridge: ``ToolExecutionContext`` is a Pydantic model
that travels on ``ToolExecutionRequest`` (surviving the JSON round-trip to the
Celery tools worker). At rebuild time it is overlaid onto any Pydantic config
that exposes matching fields — no engine-side dict munging, no knowledge of a
specific config shape.

Add a field here when a new request-scoped value must reach tool execution
(e.g. system settings, headers). The engine builds the context once; every tool
rebuild picks it up generically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from private_gpt.components.sandbox.mount import Mount

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.models.chat_state import ChatState


class ToolExecutionContext(BaseModel):
    """Snapshot of request-scoped values needed when rebuilding a server tool.

    ``mounts`` is the current mount set (requested mounts + loaded skills).
    The model is JSON-serializable so it rides the Celery tools worker payload.
    """

    mounts: list[Mount] = Field(default_factory=list)

    # -- Construction --------------------------------------------------------

    @classmethod
    def from_state(cls, state: ChatState | None) -> ToolExecutionContext | None:
        """Build from live execution state, or ``None`` when unavailable."""
        if state is None:
            return None
        # ``request.context`` is typed as the base ``ContextConfig`` but at
        # runtime is a ``ResolvedContextConfig`` carrying ``mounts``.
        mounts = getattr(state.input.request.context, "mounts", None) or []
        if not mounts:
            return None
        return cls(mounts=list(mounts))

    # -- Application ---------------------------------------------------------

    def overlay_on(self, config: BaseModel) -> BaseModel:
        """Return a copy of *config* with this context's fields applied.

        Only fields the config actually exposes are overlaid, so this stays
        generic across tool configs and forward-compatible as new fields are
        added here.
        """
        updates: dict[str, object] = {}
        if self.mounts and hasattr(config, "mounts"):
            updates["mounts"] = self.mounts
        return config.model_copy(update=updates)
