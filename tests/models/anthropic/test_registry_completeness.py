import typing
from typing import Any

import anthropic.types as sdk_types
import pytest

from tests.models.anthropic.registry import (
    ALL_MAPPINGS,
    CONTENT_BLOCK_REGISTRY,
    DELTA_REGISTRY,
    STREAMING_EVENT_REGISTRY,
    TypeMapping,
)

_IGNORED_SDK_CONTENT_BLOCK_TYPES: set[type] = {
    sdk_types.WebSearchToolResultBlock,
    sdk_types.WebFetchToolResultBlock,
    sdk_types.CodeExecutionToolResultBlock,
    sdk_types.BashCodeExecutionToolResultBlock,
    sdk_types.TextEditorCodeExecutionToolResultBlock,
    sdk_types.ToolSearchToolResultBlock,
}


def _sdk_content_block_types() -> list[type]:
    """Types that the SDK allows inside ``Message.content``."""
    annotation = sdk_types.RawContentBlockStartEvent.model_fields[
        "content_block"
    ].annotation
    return list(typing.get_args(annotation))


def _sdk_delta_types() -> list[type]:
    """Types that the SDK allows as ``RawContentBlockDeltaEvent.delta``."""
    annotation = sdk_types.RawContentBlockDeltaEvent.model_fields["delta"].annotation
    return list(typing.get_args(annotation))


def _sdk_streaming_event_types() -> list[type]:
    """Types that the SDK allows as ``RawMessageStreamEvent`` members."""
    outer = sdk_types.RawMessageStreamEvent
    # Annotated[Union[...], PropertyInfo] → unwrap once
    inner_union = typing.get_args(typing.get_args(outer)[0])
    return [t for t in inner_union if isinstance(t, type)]


def _registry_sdk_types(registry: list[TypeMapping]) -> set[type]:
    return {m.sdk_type for m in registry}


class TestRegistryCompleteness:
    def test_all_sdk_content_block_types_are_registered(self) -> None:
        sdk_types_found = _sdk_content_block_types()
        registered = _registry_sdk_types(CONTENT_BLOCK_REGISTRY)

        unregistered = [
            t
            for t in sdk_types_found
            if t not in registered and t not in _IGNORED_SDK_CONTENT_BLOCK_TYPES
        ]
        extra = [
            m.sdk_type
            for m in CONTENT_BLOCK_REGISTRY
            if m.sdk_type not in sdk_types_found
        ]

        messages: list[str] = []
        if unregistered:
            names = ", ".join(t.__name__ for t in unregistered)
            messages.append(
                f"New SDK content block type(s) detected: {names}. "
                "Add TypeMapping entries to CONTENT_BLOCK_REGISTRY in registry.py "
                "and implement corresponding models in models.py."
            )
        if extra:
            names = ", ".join(t.__name__ for t in extra)
            messages.append(
                f"SDK content block type(s) no longer exist: {names}. "
                "Remove or deprecate the TypeMapping entries and "
                "corresponding models."
            )

        assert not messages, "\n".join(messages)

    def test_all_sdk_delta_types_are_registered(self) -> None:
        sdk_types_found = _sdk_delta_types()
        registered = _registry_sdk_types(DELTA_REGISTRY)

        unregistered = [t for t in sdk_types_found if t not in registered]
        extra = [
            m.sdk_type for m in DELTA_REGISTRY if m.sdk_type not in sdk_types_found
        ]

        messages: list[str] = []
        if unregistered:
            names = ", ".join(t.__name__ for t in unregistered)
            messages.append(
                f"New SDK delta type(s) detected: {names}. "
                "Add TypeMapping entries to DELTA_REGISTRY in registry.py "
                "and implement corresponding models in models.py."
            )
        if extra:
            names = ", ".join(t.__name__ for t in extra)
            messages.append(
                f"SDK delta type(s) no longer exist: {names}. "
                "Remove or deprecate the TypeMapping entries."
            )

        assert not messages, "\n".join(messages)

    def test_all_sdk_streaming_event_types_are_registered(self) -> None:
        sdk_types_found = _sdk_streaming_event_types()
        registered = _registry_sdk_types(STREAMING_EVENT_REGISTRY)

        unregistered = [t for t in sdk_types_found if t not in registered]
        extra = [
            m.sdk_type
            for m in STREAMING_EVENT_REGISTRY
            if m.sdk_type not in sdk_types_found
        ]

        messages: list[str] = []
        if unregistered:
            names = ", ".join(t.__name__ for t in unregistered)
            messages.append(
                f"New SDK streaming event type(s) detected: {names}. "
                "Add TypeMapping entries to STREAMING_EVENT_REGISTRY in registry.py "
                "and implement corresponding models in models.py."
            )
        if extra:
            names = ", ".join(t.__name__ for t in extra)
            messages.append(
                f"SDK streaming event type(s) no longer exist: {names}. "
                "Remove or deprecate the TypeMapping entries."
            )

        assert not messages, "\n".join(messages)

    def test_all_implemented_types_have_matching_type_literal(self) -> None:
        failures: list[str] = []

        for mapping in ALL_MAPPINGS:
            if mapping.our_type is None:
                continue

            sdk_field = (
                mapping.sdk_type.model_fields.get("type") if mapping.sdk_type else None
            )
            our_field = (
                mapping.our_type.model_fields.get("type") if mapping.our_type else None
            )
            if sdk_field is None or our_field is None:
                continue

            def _literal_values(field_info: Any) -> set[str]:
                args = typing.get_args(field_info.annotation)
                return set(args) if args else set()

            sdk_literals = _literal_values(sdk_field)
            our_literals = _literal_values(our_field)

            if sdk_literals and our_literals and sdk_literals != our_literals:
                failures.append(
                    f"[{mapping.our_type.__name__}] type literal mismatch: "
                    f"SDK={sdk_literals} ours={our_literals}"
                )

        assert not failures, "\n".join(failures)

    def test_no_unimplemented_types_are_silently_ignored(self) -> None:
        missing = [m for m in ALL_MAPPINGS if m.our_type is None]
        if missing:
            details = "\n".join(
                f"  - {m.sdk_type.__name__}: {m.notes or 'no notes'}" for m in missing
            )
            pytest.fail(
                f"{len(missing)} SDK type(s) have no Zylon implementation:\n"
                f"{details}\n\n"
                "Implement the model(s) in private_gpt/chat/models.py, then "
                "update the registry entry to set our_type."
            )
