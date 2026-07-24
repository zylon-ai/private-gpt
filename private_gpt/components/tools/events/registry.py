from typing import TYPE_CHECKING

from private_gpt.components.tools.events.adapters import (
    BashCodeExecutionEventAdapter,
    ClientToolEventAdapter,
    ServerToolEventAdapter,
    TextEditorCodeExecutionEventAdapter,
    ToolEventAdapter,
)

if TYPE_CHECKING:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec


_ADAPTERS: dict[str, ToolEventAdapter] = {
    "code_execution.bash": BashCodeExecutionEventAdapter(),
    "code_execution.text_editor": TextEditorCodeExecutionEventAdapter(),
}
_DEFAULT_CLIENT_ADAPTER = ClientToolEventAdapter()
_DEFAULT_SERVER_ADAPTER = ServerToolEventAdapter()


def register_tool_event_adapter(key: str, adapter: ToolEventAdapter) -> None:
    if key in _ADAPTERS:
        raise ValueError(f"Tool event adapter {key!r} is already registered")
    _ADAPTERS[key] = adapter


def resolve_tool_event_adapter(tool_spec: "ToolSpec") -> ToolEventAdapter:
    if tool_spec.event_adapter_key is not None:
        try:
            return _ADAPTERS[tool_spec.event_adapter_key]
        except KeyError as error:
            raise ValueError(
                f"Unknown tool event adapter {tool_spec.event_adapter_key!r} "
                f"for tool {tool_spec.name!r}"
            ) from error
    if tool_spec.runtime == "server":
        return _DEFAULT_SERVER_ADAPTER
    return _DEFAULT_CLIENT_ADAPTER
