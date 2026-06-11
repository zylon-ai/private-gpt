from __future__ import annotations

from private_gpt.components.sandbox.mount import SandboxMountSpec


class SessionMountDef(SandboxMountSpec):
    """One entry of the session filesystem layout.

    ``name`` doubles as the volume name and the host subdirectory under
    ``{sessions}/{session_id}/``.
    """

    name: str


# The canonical session filesystem layout, defined exactly once. Environments
# with a different layout pass their own tuple to the Mounter.
DEFAULT_SESSION_LAYOUT: tuple[SessionMountDef, ...] = (
    SessionMountDef(name="workspace", canonical="/home/agent/", writable=True),
    SessionMountDef(
        name="uploads", canonical="/mnt/user-data/uploads/", writable=False
    ),
    SessionMountDef(name="outputs", canonical="/mnt/user-data/outputs/", writable=True),
)
