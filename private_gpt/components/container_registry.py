from __future__ import annotations

from injector import singleton


@singleton
class ContainerRegistry:
    """Track active persistent container sessions across subsystems.

    Any component that promotes a persistent sandbox session registers here.
    The chat loop queries this registry to emit a Container handle to the
    client without knowing which subsystem owns the session.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, int] = {}  # session_id → ttl_seconds

    def register(self, session_id: str, ttl_seconds: int) -> None:
        self._sessions[session_id] = ttl_seconds

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_ttl(self, session_id: str) -> int | None:
        return self._sessions.get(session_id)
