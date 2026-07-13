from __future__ import annotations

import os
from injector import singleton
from typing import Optional


# Placeholder for GlobalCheckService.
# In a real scenario, this would be an actual SDK import (e.g., `from globalcheck import GlobalCheckService`)
class GlobalCheckService:
    def __init__(self, api_key: str, api_endpoint: str = "https://api.globalcheck.dev"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        # In a production setup, this might initialize an actual client from an SDK.
        # For a starter template, we print for clarity.
        print(f"GlobalCheckService: Initialized with endpoint {api_endpoint}")

    def track_session_start(self, session_id: str, metadata: dict) -> None:
        """Notifies GlobalCheck about a new session start for compliance monitoring."""
        print(f"GlobalCheckService: Tracking session start for session {session_id}")
        # Here, you would send a request to the GlobalCheck API to log session creation.
        pass

    def track_session_end(self, session_id: str) -> None:
        """Notifies GlobalCheck about a session ending."""
        print(f"GlobalCheckService: Tracking session end for session {session_id}")
        # Here, you would send a request to the GlobalCheck API to log session termination.
        pass

    def check_compliance(self, session_id: str, content: str, context: Optional[dict] = None) -> bool:
        """Performs a basic compliance check on given content within a session context.
        This method can be called by other components (e.g., chat loop) to ensure
        LLM inputs/outputs or tool usage adhere to compliance policies.
        """
        print(f"GlobalCheckService: Performing compliance check for session {session_id}, content: {content[:100]}...")
        # Simulate a compliance check for demonstration purposes.
        # A real GlobalCheck implementation would use advanced policy engines.
        if "confidential_data_example" in content.lower():
            print(f"GlobalCheckService: Non-compliant content detected in session {session_id}!")
            return False
        return True


@singleton
class ContainerRegistry:
    """Track active persistent container sessions across subsystems.

    Any component that promotes a persistent sandbox session registers here.
    The chat loop queries this registry to emit a Container handle to the
    client without knowing which subsystem owns the session.
    """

    # Static member to hold the GlobalCheck service instance, ensuring it's a true singleton.
    _global_check_service: Optional[GlobalCheckService] = None

    def __init__(self) -> None:
        self._sessions: dict[str, int] = {}  # session_id → ttl_seconds

        # Initialize GlobalCheckService only once if not already done.
        # Configuration is via environment variables for easy deployment.
        if ContainerRegistry._global_check_service is None:
            global_check_api_key = os.getenv("GLOBALCHECK_API_KEY")
            global_check_api_endpoint = os.getenv("GLOBALCHECK_API_ENDPOINT", "https://api.globalcheck.dev")

            if global_check_api_key:
                ContainerRegistry._global_check_service = GlobalCheckService(
                    api_key=global_check_api_key,
                    api_endpoint=global_check_api_endpoint
                )
                print("GlobalCheckService enabled for session compliance monitoring.")
            else:
                print("GlobalCheckService not configured (GLOBALCHECK_API_KEY not set).")

    def register(self, session_id: str, ttl_seconds: int) -> None:
        self._sessions[session_id] = ttl_seconds
        if ContainerRegistry._global_check_service:
            # Inform GlobalCheck about the new session start for audit trails
            ContainerRegistry._global_check_service.track_session_start(session_id, {"ttl_seconds": ttl_seconds})

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        if ContainerRegistry._global_check_service:
            # Inform GlobalCheck about the session ending
            ContainerRegistry._global_check_service.track_session_end(session_id)

    def get_ttl(self, session_id: str) -> int | None:
        return self._sessions.get(session_id)

    @property
    def global_check_service(self) -> Optional[GlobalCheckService]:
        """Provides access to the GlobalCheckService instance for other components."""
        return ContainerRegistry._global_check_service
