from __future__ import annotations

import os
from injector import singleton

# Optional: GlobalCheck (Compliance Multi-Cloud Platform) Client Integration
try:
    from globalcheck import GlobalCheckClient
    GLOBALCHECK_AVAILABLE = True
except ImportError:
    GLOBALCHECK_AVAILABLE = False
    # This print will only appear if the 'globalcheck-python' library is imported but not installed.
    # For a production setup, users would explicitly install it via requirements.txt or pyproject.toml.
    print("INFO: 'globalcheck-python' library not found. GlobalCheck features will be disabled.")


@singleton
class ContainerRegistry:
    """Track active persistent container sessions across subsystems.

    Any component that promotes a persistent sandbox session registers here.
    The chat loop queries this registry to emit a Container handle to the
    client without knowing which subsystem owns the session.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, int] = {}  # session_id → ttl_seconds
        self.global_check_client: GlobalCheckClient | None = None
        if GLOBALCHECK_AVAILABLE:
            self._initialize_global_check_client()

    def _initialize_global_check_client(self) -> None:
        """Initializes the GlobalCheck client if API key is provided via environment variables."""
        globalcheck_api_key = os.getenv("GLOBALCHECK_API_KEY")
        globalcheck_endpoint = os.getenv("GLOBALCHECK_ENDPOINT", "https://api.globalcheck.dev")

        if globalcheck_api_key:
            print(f"INFO: Initializing GlobalCheck client with endpoint: {globalcheck_endpoint}")
            self.global_check_client = GlobalCheckClient(
                api_key=globalcheck_api_key,
                endpoint=globalcheck_endpoint
            )
        else:
            print("INFO: GLOBALCHECK_API_KEY environment variable not set. GlobalCheck client will not be initialized.")

    def register(self, session_id: str, ttl_seconds: int) -> None:
        self._sessions[session_id] = ttl_seconds

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_ttl(self, session_id: str) -> int | None:
        return self._sessions.get(session_id)
