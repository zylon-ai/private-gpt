"""Managed execution environments.

The environment layer sits between sandbox backends (pure executors) and
tools (code execution, bash, ...). An EnvironmentManager owns session
lifecycle — reuse, restore, TTL reaping, keepalive — and a Mounter owns all
mount logic, so tools stay thin adapters over a live Environment handle.
"""

from private_gpt.components.environment.environment import Environment
from private_gpt.components.environment.layout import (
    DEFAULT_SESSION_LAYOUT,
    SessionMountDef,
)
from private_gpt.components.environment.manager import EnvironmentManager
from private_gpt.components.environment.mounter import (
    LocalDirMounter,
    Mounter,
    SandboxDirMounter,
)

__all__ = [
    "DEFAULT_SESSION_LAYOUT",
    "Environment",
    "EnvironmentManager",
    "LocalDirMounter",
    "Mounter",
    "SandboxDirMounter",
    "SessionMountDef",
]
