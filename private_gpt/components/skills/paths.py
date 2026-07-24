from pathlib import PurePosixPath

from private_gpt.components.skills.errors import SkillDomainError, SkillErrorCode

# The canonical skills mount root inside execution environments — defined once.
SKILLS_MOUNT_ROOT = "/mnt/skills/"


def skill_path(collection: str, skill_id: str, version_id: str) -> str:
    return f"skills/{collection}/{skill_id}/{version_id}"


def skill_mount_path(skill_name: str) -> str:
    """Canonical directory a skill is mounted at, e.g. ``/mnt/skills/pdf/``."""
    return f"{SKILLS_MOUNT_ROOT}{skill_name}/"


def normalize_skill_relative_path(path: str) -> str:
    """Normalize a skill-relative path and reject absolute / traversal paths."""
    normalized = path.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if parsed.is_absolute():
        raise SkillDomainError(
            SkillErrorCode.UNSAFE_PATH_ABSOLUTE,
            f"Unsafe file path (absolute): {path!r}",
            params={"path": path},
        )
    if ".." in parsed.parts:
        raise SkillDomainError(
            SkillErrorCode.UNSAFE_PATH_TRAVERSAL,
            f"Unsafe file path (path traversal): {path!r}",
            params={"path": path},
        )
    parts = list(parsed.parts)
    if not parts or parts == [""]:
        raise SkillDomainError(
            SkillErrorCode.UNSAFE_PATH_ABSOLUTE,
            f"Unsafe file path (empty): {path!r}",
            params={"path": path},
        )
    if parts[-1].lower() == "skill.md":
        parts[-1] = "SKILL.md"
    return "/".join(parts)
