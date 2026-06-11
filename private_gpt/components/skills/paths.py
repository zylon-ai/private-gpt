# The canonical skills mount root inside execution environments — defined once.
SKILLS_MOUNT_ROOT = "/mnt/skills/"


def skill_path(collection: str, skill_id: str, version_id: str) -> str:
    return f"skills/{collection}/{skill_id}/{version_id}"


def skill_mount_path(skill_id: str) -> str:
    """Canonical directory a skill is mounted at, e.g. ``/mnt/skills/pdf/``."""
    return f"{SKILLS_MOUNT_ROOT}{skill_id}/"
