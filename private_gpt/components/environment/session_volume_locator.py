from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from injector import inject, singleton

from private_gpt.settings.settings import Settings


@singleton
class SessionVolumeLocator:
    """Injectable access point for session host-paths.

    Reads volume_root and vfs_sessions_prefix from CodeExecutionSettings so
    the path construction lives in exactly one place.  Raises HTTP 503 on any
    path request when volume_root is not configured.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        cfg = settings.code_execution
        if cfg.volume_root is not None:
            self._sessions: Path | None = (
                Path(cfg.volume_root) / cfg.vfs_sessions_prefix
            )
        else:
            self._sessions = None

    def _require(self) -> Path:
        if self._sessions is None:
            raise HTTPException(
                status_code=503,
                detail="Files API requires code_execution.volume_root to be configured.",
            )
        return self._sessions

    def uploads_path(self, session_id: str) -> Path:
        return self._require() / session_id / "uploads"

    def outputs_path(self, session_id: str) -> Path:
        return self._require() / session_id / "outputs"
