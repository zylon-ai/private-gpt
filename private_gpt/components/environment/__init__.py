"""Managed execution environments.

The environment layer sits between sandbox backends (pure executors) and
tools (code execution, bash, ...). An EnvironmentManager owns session
lifecycle — reuse, restore, TTL reaping, keepalive — a LayoutMounter owns
the session filesystem structure, and a list of ContentMounters handles how
bundle content (skills, tools, ...) reaches the sandbox.
"""

from private_gpt.components.environment.content_mounter import (
    ContentMounter,
    FetchContentMounter,
    InlineContentMounter,
    LocalStorageContentMounter,
)
from private_gpt.components.environment.environment import Environment
from private_gpt.components.environment.layout import (
    DEFAULT_SESSION_LAYOUT,
    SessionMountDef,
)
from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.environment.mounter import (
    LayoutMounter,
    LocalDirMounter,
    Mounter,
    SandboxDirMounter,
)

__all__ = [
    "DEFAULT_SESSION_LAYOUT",
    "ContentMounter",
    "Environment",
    "EnvironmentManager",
    "FetchContentMounter",
    "InlineContentMounter",
    "LayoutMounter",
    "LocalDirMounter",
    "LocalStorageContentMounter",
    "Mounter",
    "SandboxDirMounter",
    "SessionMountDef",
]
