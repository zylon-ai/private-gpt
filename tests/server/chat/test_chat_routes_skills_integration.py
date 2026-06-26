import json
import uuid
from typing import Any

import pytest
from httpx import AsyncClient
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import (
    Message,
    TextBlock,
    ToolResultBlock,
)
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.utils.artifact_input import SkillArtifact
from private_gpt.settings.settings import Settings
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector


class SkillChatCapture:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.tool_names_per_call: list[list[str]] = []


async def mock_llm_with_capture(
    injector: MockInjector,
    capture: SkillChatCapture,
    deltas: list[list[str | ToolSelection]] | None = None,
) -> None:
    deltas = [["ok"]] * 20 if deltas is None else [*deltas, *([["done"]] * 20)]
    mock_llm = get_mock_function_calling_llm(deltas)
    original = mock_llm.astream_chat_with_tools

    async def capturing_astream(
        tools: Any,
        user_msg: Any = None,
        chat_history: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> Any:
        names: list[str] = []
        for tool in tools or []:
            metadata = getattr(tool, "metadata", None)
            name = getattr(metadata, "name", None) or getattr(tool, "name", None)
            if name:
                names.append(str(name))
        capture.tool_names_per_call.append(names)

        if chat_history:
            system_messages = [m for m in chat_history if m.role.value == "system"]
            capture.system_prompts.append(
                "\n".join([m.content or "" for m in system_messages])
            )
        else:
            capture.system_prompts.append("")

        gen = await original(tools, user_msg, chat_history, **kwargs)
        async for response in gen:
            yield response

    async def coro(*args: Any, **kwargs: Any) -> Any:
        return capturing_astream(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro
    llm_component = injector.get(LLMComponent)
    llm_component.llm = mock_llm
    injector.bind_mock(LLMComponent, mock_llm)


def _skill_md(name: str, description: str, body: str) -> bytes:
    return f'---\nname: {name}\ndescription: "{description}"\n---\n\n{body}\n'.encode()


async def _create_skill(
    async_test_client: AsyncClient,
    *,
    collection: str,
    name: str,
    loading: str = "lazy",
    body: str | None = None,
) -> None:
    response = await async_test_client.post(
        "/v1/skills",
        data={
            "display_title": name,
            "collection": collection,
            "loading": loading,
        },
        files=[
            (
                "files",
                (
                    "SKILL.md",
                    _skill_md(
                        name=name,
                        description=f"{name} description",
                        body=body or f"{name} body",
                    ),
                    "text/markdown",
                ),
            )
        ],
    )
    assert response.status_code == 200


def _skill_tools(*, with_deferred_custom: bool = False) -> list[dict[str, Any]]:
    tools = [
        {"name": "load_skill", "type": "load_skill_v1"},
        {"name": "unload_skill", "type": "unload_skill_v1"},
        {"name": "list_skills", "type": "list_skills_v1"},
    ]
    if with_deferred_custom:
        tools.append(
            {
                "name": "delayed_custom",
                "description": "Visible only after first loaded skill",
                "input_schema": {"type": "object", "properties": {}},
                "defer_loading": True,
            }
        )
    return tools


def _assistant_load_history(skill_name: str) -> dict[str, Any]:
    payload = {
        "name": skill_name,
        "loaded": True,
        "skill_id": "dummy",
        "version": "dummy",
    }
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tu_load",
                "name": "load_skill",
                "input": {"name": skill_name},
            },
            {
                "type": "tool_result",
                "tool_use_id": "tu_load",
                "content": [{"type": "text", "text": json.dumps(payload)}],
            },
        ],
    }


def _assistant_load_history_many(skill_names: list[str]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for idx, name in enumerate(skill_names):
        tool_id = f"tu_{idx}"
        payload = {
            "name": name,
            "loaded": True,
            "skill_id": "dummy",
            "version": "dummy",
        }
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "load_skill",
                "input": {"name": name},
            }
        )
        blocks.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": json.dumps(payload)}],
            }
        )
    return {"role": "assistant", "content": blocks}


@pytest.mark.anyio
async def test_skill_tools_visible_only_when_activatable(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    capture = SkillChatCapture()
    await mock_llm_with_capture(injector, capture)

    empty_collection = str(uuid.uuid4())
    no_skills_body = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(),
        tool_context=[
            SkillArtifact(skill_filter=SkillFilter(collection=empty_collection))
        ],
    )
    no_skills_resp = await async_test_client.post(
        "/v1/messages", json=no_skills_body.model_dump()
    )
    assert no_skills_resp.status_code == 200
    assert SKILL_LOAD_TOOL_NAME not in capture.tool_names_per_call[-1]
    assert SKILL_UNLOAD_TOOL_NAME not in capture.tool_names_per_call[-1]

    active_collection = str(uuid.uuid4())
    await _create_skill(
        async_test_client, collection=active_collection, name="active-skill"
    )
    with_skills_body = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(),
        tool_context=[
            SkillArtifact(skill_filter=SkillFilter(collection=active_collection))
        ],
    )
    with_skills_resp = await async_test_client.post(
        "/v1/messages", json=with_skills_body.model_dump()
    )
    assert with_skills_resp.status_code == 200
    assert SKILL_LOAD_TOOL_NAME in capture.tool_names_per_call[-1]
    assert SKILL_UNLOAD_TOOL_NAME not in capture.tool_names_per_call[-1]

    # Once a skill is loaded unload_skill becomes visible and load_skill disappears
    loaded_body = {
        "messages": [
            _assistant_load_history("active-skill"),
            {"role": "user", "content": "now what"},
        ],
        "tools": _skill_tools(),
        "tool_context": [
            {"type": "skill", "skill_filter": {"collection": active_collection}}
        ],
    }
    loaded_resp = await async_test_client.post("/v1/messages", json=loaded_body)
    assert loaded_resp.status_code == 200
    assert SKILL_UNLOAD_TOOL_NAME in capture.tool_names_per_call[-1]
    assert SKILL_LOAD_TOOL_NAME not in capture.tool_names_per_call[-1]


@pytest.mark.anyio
async def test_loaded_skill_disappears_from_catalog(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(async_test_client, collection=collection, name="catalog-skill")

    settings = injector.get(Settings)
    previous_mode = settings.skills.skill_injection_mode
    settings.skills.skill_injection_mode = "system_prompt"
    try:
        capture = SkillChatCapture()
        await mock_llm_with_capture(injector, capture)

        not_loaded = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        not_loaded_resp = await async_test_client.post(
            "/v1/messages", json=not_loaded.model_dump()
        )
        assert not_loaded_resp.status_code == 200
        assert "<available_skills>" in capture.system_prompts[-1]

        loaded_body = {
            "messages": [
                _assistant_load_history("catalog-skill"),
                {"role": "user", "content": "hello again"},
            ],
            "tools": _skill_tools(),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        loaded_resp = await async_test_client.post("/v1/messages", json=loaded_body)
        assert loaded_resp.status_code == 200
        assert "<available_skills>" not in capture.system_prompts[-1]
    finally:
        settings.skills.skill_injection_mode = previous_mode


@pytest.mark.anyio
async def test_defer_loading_hidden_until_first_skill_loaded(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(async_test_client, collection=collection, name="defer-skill")

    capture = SkillChatCapture()
    await mock_llm_with_capture(injector, capture)

    before = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(with_deferred_custom=True),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection=collection))],
    )
    before_resp = await async_test_client.post("/v1/messages", json=before.model_dump())
    assert before_resp.status_code == 200
    assert "delayed_custom" not in capture.tool_names_per_call[-1]

    after = {
        "messages": [
            _assistant_load_history("defer-skill"),
            {"role": "user", "content": "next"},
        ],
        "tools": _skill_tools(with_deferred_custom=True),
        "tool_context": [{"type": "skill", "skill_filter": {"collection": collection}}],
    }
    after_resp = await async_test_client.post("/v1/messages", json=after)
    assert after_resp.status_code == 200
    assert "delayed_custom" in capture.tool_names_per_call[-1]


@pytest.mark.anyio
async def test_skill_injection_mode_system_prompt_loads_lazy_body(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    sentinel = "SYSTEM_PROMPT_SENTINEL"
    await _create_skill(
        async_test_client,
        collection=collection,
        name="system-mode-skill",
        loading="lazy",
        body=sentinel,
    )

    settings = injector.get(Settings)
    previous_mode = settings.skills.skill_injection_mode
    settings.skills.skill_injection_mode = "system_prompt"
    try:
        capture = SkillChatCapture()
        await mock_llm_with_capture(injector, capture)

        body = {
            "messages": [
                _assistant_load_history("system-mode-skill"),
                {"role": "user", "content": "apply loaded skill"},
            ],
            "tools": _skill_tools(),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        response = await async_test_client.post("/v1/messages", json=body)
        assert response.status_code == 200
        assert sentinel in capture.system_prompts[-1]
    finally:
        settings.skills.skill_injection_mode = previous_mode


@pytest.mark.anyio
async def test_skill_injection_mode_tool_result_includes_full_body_only_in_tool_result(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    sentinel = "TOOL_RESULT_SENTINEL"
    await _create_skill(
        async_test_client,
        collection=collection,
        name="tool-result-skill",
        loading="lazy",
        body=sentinel,
    )

    settings = injector.get(Settings)
    previous_mode = settings.skills.skill_injection_mode
    settings.skills.skill_injection_mode = "tool_result"
    try:
        capture = SkillChatCapture()
        deltas = [
            [
                ToolSelection(
                    tool_id="load_skill",
                    tool_name="load_skill",
                    tool_kwargs={"name": "tool-result-skill"},
                )
            ],
            ["done"],
        ]
        await mock_llm_with_capture(injector, capture, deltas=deltas)

        first_body = ChatBody(
            messages=[MessageInput(content="load now", role="user")],
            tools=[{"name": "load_skill", "type": "load_skill_v1"}],
            tool_choice={"type": "tool", "name": "load_skill"},
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        first_response = await async_test_client.post(
            "/v1/messages", json=first_body.model_dump()
        )
        assert first_response.status_code == 200
        completion = Message.model_validate(first_response.json())
        tool_result_blocks = [
            block for block in completion.content if isinstance(block, ToolResultBlock)
        ]
        assert tool_result_blocks
        text_blocks = [
            block
            for block in tool_result_blocks[0].content
            if isinstance(block, TextBlock)
        ]
        payload = json.loads(text_blocks[0].text)
        assert sentinel in payload["instructions"]

        second_body = {
            "messages": [
                _assistant_load_history("tool-result-skill"),
                {"role": "user", "content": "next turn"},
            ],
            "tools": _skill_tools(),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        second_response = await async_test_client.post("/v1/messages", json=second_body)
        assert second_response.status_code == 200
        assert sentinel not in capture.system_prompts[-1]
    finally:
        settings.skills.skill_injection_mode = previous_mode


@pytest.mark.anyio
async def test_maximum_loaded_skills_evicts_oldest_loaded_skill(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(
        async_test_client, collection=collection, name="alpha", body="BODY_ALPHA"
    )
    await _create_skill(
        async_test_client, collection=collection, name="beta", body="BODY_BETA"
    )
    await _create_skill(
        async_test_client, collection=collection, name="gamma", body="BODY_GAMMA"
    )

    settings = injector.get(Settings)
    previous_mode = settings.skills.skill_injection_mode
    settings.skills.skill_injection_mode = "system_prompt"
    try:
        capture = SkillChatCapture()
        await mock_llm_with_capture(injector, capture)

        body = {
            "messages": [
                _assistant_load_history_many(["alpha", "beta", "gamma"]),
                {"role": "user", "content": "apply active skills"},
            ],
            "tools": _skill_tools(),
            "maximum_loaded_skills": 2,
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        response = await async_test_client.post("/v1/messages", json=body)
        assert response.status_code == 200
        prompt = capture.system_prompts[-1]
        assert "BODY_BETA" in prompt
        assert "BODY_GAMMA" in prompt
        assert "BODY_ALPHA" not in prompt
    finally:
        settings.skills.skill_injection_mode = previous_mode
