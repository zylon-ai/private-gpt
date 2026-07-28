from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from private_gpt.components.tools.events.adapters import (
    ClientToolEventAdapter,
    ServerToolEventAdapter,
    ToolEventAdapter,
)
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec


def load_tool_event_adapter_class(path: str) -> type[ToolEventAdapter]:
    module_name, qualname = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    resolved: object = module
    for attribute in qualname.split("."):
        resolved = getattr(resolved, attribute)
    if not isinstance(resolved, type) or not issubclass(resolved, ToolEventAdapter):
        raise TypeError(f"{path!r} is not a ToolEventAdapter subclass")
    return resolved


def resolve_tool_event_adapter(tool_spec: ToolSpec) -> ToolEventAdapter:
    mode = settings().code_execution.tools.server_tool_result_mode
    if mode == "client":
        return ClientToolEventAdapter()
    adapter_cls = tool_spec.event_adapter
    if adapter_cls is None:
        return (
            ServerToolEventAdapter()
            if tool_spec.runtime == "server"
            else ClientToolEventAdapter()
        )
    return adapter_cls()
