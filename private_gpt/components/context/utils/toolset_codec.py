"""Serialize and deserialize tool specs for context layers."""

import json

from private_gpt.components.chat.models.chat_config_models import ToolSpec

_TOOL_SPEC_KEYS = ("name", "type", "description", "input_schema")


def serialize_tool_specs(tool_specs: list[ToolSpec]) -> str:
    """Serialize tool specs into compact JSON for context storage."""
    payload: list[dict[str, object]] = []
    for tool in tool_specs:
        item: dict[str, object] = {}
        for key in _TOOL_SPEC_KEYS:
            value = getattr(tool, key)
            if value is not None:
                item[key] = value
        payload.append(item)
    return json.dumps(payload)


def deserialize_tool_specs(raw_content: str) -> list[ToolSpec]:
    """Deserialize TOOL_DEFINITIONS layer content into tool specs."""
    payload = json.loads(raw_content)
    if not isinstance(payload, list):
        return []
    valid_payload = [item for item in payload if isinstance(item, dict)]
    return [ToolSpec.model_validate(item) for item in valid_payload]
