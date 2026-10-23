from __future__ import annotations

from private_gpt.components.sandbox.mount import Mount


class SessionMountDef(Mount):
    """One entry of the session filesystem layout.

    ``name`` doubles as the volume name and the host subdirectory under
    ``{sessions}/{session_id}/``.
    """

    name: str


# The canonical session filesystem layout, defined exactly once. Environments
# with a different layout pass their own tuple to the Mounter.
DEFAULT_SESSION_LAYOUT: tuple[SessionMountDef, ...] = (
    SessionMountDef(
        name="user",
        target="/home/agent/workspace/",
        access="rw",
        description="Working directory — create all new files here",
    ),
    SessionMountDef(
        name="uploads",
        target="/mnt/user-data/uploads/",
        access="ro",
        description="Files uploaded by the user",
    ),
    SessionMountDef(
        name="outputs",
        target="/mnt/user-data/outputs/",
        access="rw",
        description="Deliverables the user can download",
    ),
)


def canonical_to_storage_path(
    canonical: str, layout: tuple[SessionMountDef, ...] = DEFAULT_SESSION_LAYOUT
) -> str:
    """Map a canonical sandbox path to its storage path (e.g. ``user/foo.txt``)."""
    for mount in layout:
        if canonical.startswith(mount.target):
            relative = canonical[len(mount.target) :]
            return f"{mount.name}/{relative}"
    return canonical


def storage_to_canonical_path(
    storage: str, layout: tuple[SessionMountDef, ...] = DEFAULT_SESSION_LAYOUT
) -> str:
    """Map a storage path (e.g. ``user/foo.txt``) back to its canonical form."""
    for mount in layout:
        prefix = f"{mount.name}/"
        if storage.startswith(prefix):
            relative = storage[len(prefix) :]
            return f"{mount.target}{relative}"
    return storage
