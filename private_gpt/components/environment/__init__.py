"""Managed execution environments.

The environment layer sits between sandbox backends (pure executors) and
tools (code execution, bash, ...). An EnvironmentManager owns session
lifecycle — reuse, restore, TTL reaping, keepalive — a LayoutMounter owns
the session filesystem structure, and a HydratingEnvironmentManager (dev
only) syncs namespace content into the volume roots before acquire. Every
mount is a bind volume; nothing is materialized into a running sandbox.
"""

from private_gpt.components.environment.environment import Environment
from private_gpt.components.environment.hydration import HydratingEnvironmentManager
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
    "Environment",
    "EnvironmentManager",
    "HydratingEnvironmentManager",
    "LayoutMounter",
    "LocalDirMounter",
    "Mounter",
    "SandboxDirMounter",
    "SessionMountDef",
]
