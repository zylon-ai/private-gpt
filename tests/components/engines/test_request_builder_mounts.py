"""Tests for the single-mount-set merge in build_request_from_context_stack."""

from __future__ import annotations

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import MountsLayer
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.utils.request_builder import (
    build_request_from_context_stack,
)
from private_gpt.components.sandbox.mount import Mount, MountSource
from private_gpt.components.tools.types import ToolValidationMode


def _mount(target: str, *, source: str | None = None) -> Mount:
    return Mount(
        name=f"mount:{target}",
        target=target,
        access="rw",
        source=MountSource(namespace=source or "", path=""),
    )


def _request(mounts: list[Mount]) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(
            model="contract-model",
            prompt="Contract system prompt",
            blob_visibility=BlobVisibilityMode.INTERNAL,
        ),
        tool_config=ResolvedToolConfig(
            tools=[ToolSpec(name="bash", type="bash_v1")],
            validation_mode=ToolValidationMode.EAGER,
        ),
        context=ResolvedContextConfig(
            correlation_id="contract-correlation",
            maximum_context_length=98_765,
            mounts=mounts,
        ),
    )


def test_mount_plan_volumes_survive_and_stack_mounts_are_appended() -> None:
    """Backend mount-plan volumes + skill mounts end up in one mount set."""
    plan = _mount("/mnt/artifacts/org-1/", source="/host/artifacts/org-1")
    stack = ContextStack().append_layer(
        MountsLayer(
            mounts=[_mount("/mnt/skills/pdf/")],
            source="skills",
        )
    )

    request = build_request_from_context_stack(_request([plan]), stack)

    assert [(m.target, m.access) for m in request.context.mounts] == [
        ("/mnt/artifacts/org-1/", "rw"),
        ("/mnt/skills/pdf/", "rw"),
    ]


def test_merge_is_idempotent_across_rebuilds() -> None:
    """build_request runs per iteration; mounts must not accumulate."""
    plan = _mount("/mnt/artifacts/org-1/", source="/host/artifacts/org-1")
    stack = ContextStack().append_layer(
        MountsLayer(mounts=[_mount("/mnt/skills/pdf/")], source="skills")
    )

    first = build_request_from_context_stack(_request([plan]), stack)
    second = build_request_from_context_stack(first, stack)
    third = build_request_from_context_stack(second, stack)

    assert len(third.context.mounts) == 2
    assert [(m.target, m.access) for m in third.context.mounts] == [
        ("/mnt/artifacts/org-1/", "rw"),
        ("/mnt/skills/pdf/", "rw"),
    ]
