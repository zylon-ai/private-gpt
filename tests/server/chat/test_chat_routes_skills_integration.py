import json
import uuid
from typing import Any
from unittest.mock import Mock

import pytest
from httpx import AsyncClient
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
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
        self.histories: list[list[ChatMessage]] = []


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
        capture.histories.append(list(chat_history or []))

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
    llm_component.get_llm = Mock(return_value=mock_llm)
    injector.bind_mock(LLMComponent, llm_component)


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


def _assistant_server_tool_load_history(skill_name: str) -> dict[str, Any]:
    """Client follow-up history: list_skills + load_skill as server tool blocks."""
    list_id = "srvtoolu_fa9fc7a760314303a8e00126e973631f"
    load_id = "srvtoolu_95b5b4bca35c4db883f08d9762b9b5ee"
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll help you create a skill. Let me first check what skills are available.\n\n",
                "start_timestamp": "2026-08-19T13:47:22.448574Z",
                "stop_timestamp": "2026-08-19T13:47:22.850456Z",
            },
            {
                "type": "server_tool_use",
                "id": list_id,
                "name": "list_skills",
                "input": {"page": 0, "page_size": 20},
                "start_timestamp": "2026-08-19T13:47:22.925823Z",
                "stop_timestamp": "2026-08-19T13:47:23.336161Z",
            },
            {
                "type": "server_tool_result",
                "tool_use_id": list_id,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "skills": [
                                    {
                                        "name": skill_name,
                                        "description": f"{skill_name} description",
                                    }
                                ],
                                "page": 0,
                                "page_size": 20,
                                "total": 1,
                                "has_more": False,
                            }
                        ),
                    }
                ],
                "is_error": False,
                "start_timestamp": "2026-08-19T13:47:23.351800Z",
                "stop_timestamp": "2026-08-19T13:47:23.351871Z",
            },
            {
                "type": "text",
                "text": "Let me start by loading the skill-creator tool.\n\n",
                "start_timestamp": "2026-08-19T13:47:26.178468Z",
                "stop_timestamp": "2026-08-19T13:47:26.948111Z",
            },
            {
                "type": "server_tool_use",
                "id": load_id,
                "name": "load_skill",
                "input": {"name": skill_name},
                "start_timestamp": "2026-08-19T13:47:26.982531Z",
                "stop_timestamp": "2026-08-19T13:47:27.075400Z",
            },
            {
                "type": "server_tool_result",
                "tool_use_id": load_id,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "name": skill_name,
                                "skill_id": "dummy",
                                "version": "dummy",
                                "loaded": True,
                            }
                        ),
                    }
                ],
                "is_error": False,
                "start_timestamp": "2026-08-19T13:47:27.088033Z",
                "stop_timestamp": "2026-08-19T13:47:27.088100Z",
            },
            {
                "type": "text",
                "text": "Great! Let me help you create a skill.",
                "start_timestamp": "2026-08-19T13:47:30.751553Z",
                "stop_timestamp": "2026-08-19T13:47:36.252591Z",
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
async def test_client_followup_with_server_tool_load_keeps_skill_body(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    sentinel = "SKILL_CREATOR_BODY_SENTINEL"
    await _create_skill(
        async_test_client,
        collection=collection,
        name="skill-creator",
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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Create a skill for building internal metrics dashboards.",
                        }
                    ],
                },
                _assistant_server_tool_load_history("skill-creator"),
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you draft a version based on that?",
                        }
                    ],
                },
            ],
            "tools": [{"name": "skills", "type": "skills_v1"}],
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        response = await async_test_client.post("/v1/messages", json=body)
        assert response.status_code == 200, response.text
        prompt = capture.system_prompts[-1]
        assert sentinel in prompt
        assert "<skill_content" in prompt
        assert "<available_skills>" not in prompt
        assert SKILL_UNLOAD_TOOL_NAME in capture.tool_names_per_call[-1]
        assert SKILL_LOAD_TOOL_NAME not in capture.tool_names_per_call[-1]
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


@pytest.mark.anyio
async def test_eager_skill_loaded_without_load_skill_and_omitted_from_catalog(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(
        async_test_client,
        collection=collection,
        name="eager-guide",
        loading="eager",
        body="EAGER_SENTINEL",
    )
    await _create_skill(
        async_test_client,
        collection=collection,
        name="lazy-skill",
        loading="lazy",
        body="LAZY_SENTINEL",
    )

    settings = injector.get(Settings)
    previous_mode = settings.skills.skill_injection_mode
    settings.skills.skill_injection_mode = "system_prompt"
    try:
        capture = SkillChatCapture()
        await mock_llm_with_capture(injector, capture)

        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200
        prompt = capture.system_prompts[-1]
        assert "EAGER_SENTINEL" in prompt
        assert "LAZY_SENTINEL" not in prompt
        assert "<name>lazy-skill</name>" in prompt
        assert "<name>eager-guide</name>" not in prompt
        assert SKILL_LOAD_TOOL_NAME in capture.tool_names_per_call[-1]
        assert SKILL_LIST_TOOL_NAME in capture.tool_names_per_call[-1]
        assert SKILL_UNLOAD_TOOL_NAME not in capture.tool_names_per_call[-1]
    finally:
        settings.skills.skill_injection_mode = previous_mode


@pytest.mark.anyio
async def test_list_skills_returns_only_unloaded_lazy_skills(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(
        async_test_client,
        collection=collection,
        name="eager-guide",
        loading="eager",
        body="EAGER_SENTINEL",
    )
    await _create_skill(
        async_test_client,
        collection=collection,
        name="lazy-skill",
        loading="lazy",
        body="LAZY_SENTINEL",
    )

    capture = SkillChatCapture()
    await mock_llm_with_capture(
        injector,
        capture,
        deltas=[
            [
                ToolSelection(
                    tool_id="tu_list",
                    tool_name=SKILL_LIST_TOOL_NAME,
                    tool_kwargs={"page": 0, "page_size": 20},
                )
            ],
            ["done"],
        ],
    )

    body = ChatBody(
        messages=[MessageInput(content="what skills can I load", role="user")],
        tools=_skill_tools(),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection=collection))],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    assert len(capture.histories) >= 2

    listed_names: set[str] = set()
    for message in capture.histories[-1]:
        raw = message.content
        if not isinstance(raw, str) or "skills" not in raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("skills"), list):
            listed_names = {
                str(item["name"]) for item in payload["skills"] if "name" in item
            }
            break
    assert listed_names == {"lazy-skill"}
