from dataclasses import dataclass, field


@dataclass
class ContentBundle:
    """A named bundle of files to mount at a canonical path.

    Backend-agnostic: local provider materializes files to disk, OpenSandbox
    provider uploads them to the container. Works for skills, plugins, or any
    future content type.

    canonical_path must end with "/" (e.g. "/mnt/skills/my-skill/").
    files maps relative paths to raw bytes (e.g. {"SKILL.md": b"..."}).
    """

    canonical_path: str
    files: dict[str, bytes] = field(default_factory=dict)
    writable: bool = False
