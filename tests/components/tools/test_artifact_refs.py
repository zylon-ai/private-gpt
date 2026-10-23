from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.schema import TextNode

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.tools.artifact_refs import (
    SHAPE_ERROR,
    resolve_artifact_refs,
)
from private_gpt.components.tools.builders.semantic_search_builder import (
    SemanticSearchToolBuilder,
)
from private_gpt.components.tools.builders.tabular_data_builder import (
    TabularDataToolBuilder,
)

REPORT_2020 = "01a13d96-e5e7-7aaa-8000-000000002020"
REPORT_2024 = "01a13d9b-e5e7-7aaa-8000-000000002024"
DUP_A = "01a13d96-e5e7-7aaa-8000-00000000000a"
DUP_B = "01a13d96-e5e7-7aaa-8000-00000000000b"

FILES = {
    REPORT_2020: "2020-Annual-Report.md",
    REPORT_2024: "2024-Annual-Report.md",
    DUP_A: "notes.md",
    DUP_B: "notes.md",
}


class FakeNodeStore:
    """Minimal node store: one node per artifact, exact-match metadata filters."""

    def __init__(self, files: dict[str, str]) -> None:
        self.nodes = [
            TextNode(
                id_=f"node-{artifact_id}",
                text="x",
                metadata={"artifact_id": artifact_id, "file_name": file_name},
            )
            for artifact_id, file_name in files.items()
        ]
        self.calls: list[dict[str, Any]] = []

    def filtered_nodes(
        self,
        collection: str,
        artifacts: list[str] | None = None,
        filter_dicts: list[dict[str, Any]] | None = None,
        node_ids: list[str] | None = None,
        limit: int | None = None,
        **_: Any,
    ) -> list[TextNode]:
        self.calls.append({"artifacts": artifacts, "filter_dicts": filter_dicts})
        result = [
            node
            for node in self.nodes
            if (not artifacts or node.metadata["artifact_id"] in artifacts)
            and all(
                node.metadata.get(f["key"]) == f["value"] for f in filter_dicts or []
            )
        ]
        return result[:limit] if limit else result


@pytest.fixture
def store() -> FakeNodeStore:
    return FakeNodeStore(FILES)


def _ctx(artifacts: list[str] | None = None) -> ContextFilter:
    return ContextFilter(collection="c", artifacts=artifacts)


async def _resolve(refs: Any, store: FakeNodeStore, ctx: ContextFilter | None = None):
    return await resolve_artifact_refs(
        refs, context_filter=ctx or _ctx(), node_store_component=store
    )


async def test_matching_ref_returns_ids(store: FakeNodeStore) -> None:
    ids = await _resolve(
        [
            {"name": "2024-Annual-Report.md", "artifact_id": REPORT_2024},
            {"name": "2020-Annual-Report.md", "artifact_id": REPORT_2020},
        ],
        store,
    )
    assert ids == [REPORT_2024, REPORT_2020]


async def test_no_refs_means_no_restriction(store: FakeNodeStore) -> None:
    assert await _resolve(None, store) is None
    assert await _resolve([], store) is None


async def test_mismatch_gives_correct_id_and_real_name(store: FakeNodeStore) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [{"name": "2024-Annual-Report.md", "artifact_id": REPORT_2020}], store
        )
    message = str(exc.value)
    assert message == (
        "artifacts rejected, nothing was searched: "
        f"'2024-Annual-Report.md' has artifact_id {REPORT_2024}; "
        f"you passed {REPORT_2020}, which is '2020-Annual-Report.md'. "
        "Call again with the matching artifact_id."
    )


async def test_all_bad_refs_reported_together(store: FakeNodeStore) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [
                {"name": "2024-Annual-Report.md", "artifact_id": REPORT_2020},
                {"name": "2020-Annual-Report.md", "artifact_id": REPORT_2024},
            ],
            store,
        )
    message = str(exc.value)
    assert f"'2024-Annual-Report.md' has artifact_id {REPORT_2024}" in message
    assert f"'2020-Annual-Report.md' has artifact_id {REPORT_2020}" in message


async def test_ambiguous_name_lists_candidates(store: FakeNodeStore) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve([{"name": "notes.md", "artifact_id": REPORT_2020}], store)
    message = str(exc.value)
    assert "several files are named 'notes.md'" in message
    assert f"'notes.md' (artifact_id {DUP_A})" in message
    assert f"'notes.md' (artifact_id {DUP_B})" in message
    assert "Pick the one you mean" in message


async def test_unknown_id_with_known_name(store: FakeNodeStore) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [{"name": "2024-Annual-Report.md", "artifact_id": "missing"}], store
        )
    message = str(exc.value)
    assert "nothing was searched" in message
    assert f"'2024-Annual-Report.md' has artifact_id {REPORT_2024}" in message
    assert "you passed missing, which matches no file" in message


async def test_unknown_id_and_unknown_name(store: FakeNodeStore) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve([{"name": "ghost.pdf", "artifact_id": "missing"}], store)
    assert "no file is named 'ghost.pdf'" in str(exc.value)


async def test_out_of_scope_id_rejected(store: FakeNodeStore) -> None:
    ctx = _ctx(artifacts=[REPORT_2024])
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [{"name": "2020-Annual-Report.md", "artifact_id": REPORT_2020}],
            store,
            ctx,
        )
    message = str(exc.value)
    assert f"you passed {REPORT_2020}, which is not available here" in message
    # The name lookup stays inside the request scope
    assert "no file is named '2020-Annual-Report.md'" in message
    name_lookups = [c for c in store.calls if c["filter_dicts"]]
    assert name_lookups
    assert all(c["artifacts"] == [REPORT_2024] for c in name_lookups)


@pytest.mark.parametrize(
    "refs",
    [
        [REPORT_2024],
        REPORT_2024,
        [{"artifact_id": REPORT_2024}],
        {"name": "2024-Annual-Report.md", "artifact_id": REPORT_2024},
    ],
)
async def test_wrong_shape_rejected(store: FakeNodeStore, refs: Any) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve(refs, store)
    assert str(exc.value) == f"artifacts rejected, nothing was searched: {SHAPE_ERROR}"
    assert SHAPE_ERROR == (
        'Pass each file as {"name": ..., "artifact_id": ...}, '
        "copied from the same entry of the file list."
    )


@pytest.mark.parametrize(
    "name",
    [
        "2024-annual-report.md",
        "  2024-Annual-Report.md ",
        "reports/2024-Annual-Report.md",
        "C:\\reports\\2024-ANNUAL-REPORT.MD",
    ],
)
async def test_name_normalization(store: FakeNodeStore, name: str) -> None:
    ids = await _resolve([{"name": name, "artifact_id": REPORT_2024}], store)
    assert ids == [REPORT_2024]


async def test_mismatch_with_folder_name_finds_correct_id(
    store: FakeNodeStore,
) -> None:
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [{"name": "reports/2024-Annual-Report.md", "artifact_id": REPORT_2020}],
            store,
        )
    assert f"has artifact_id {REPORT_2024}" in str(exc.value)


async def test_case_mismatch_found_through_scope(store: FakeNodeStore) -> None:
    ctx = _ctx(artifacts=[REPORT_2020, REPORT_2024])
    with pytest.raises(ValueError) as exc:
        await _resolve(
            [{"name": "2024-ANNUAL-report.md", "artifact_id": REPORT_2020}],
            store,
            ctx,
        )
    assert f"'2024-Annual-Report.md' has artifact_id {REPORT_2024}" in str(exc.value)


def _semantic_builder(store: FakeNodeStore) -> SemanticSearchToolBuilder:
    builder = SemanticSearchToolBuilder.__new__(SemanticSearchToolBuilder)
    builder.llm_component = MagicMock()
    builder.node_store_component = store  # type: ignore[assignment]
    builder.settings = MagicMock()
    builder.prompt_builder_service = MagicMock()
    return builder


def _tabular_builder(store: FakeNodeStore) -> TabularDataToolBuilder:
    builder = TabularDataToolBuilder.__new__(TabularDataToolBuilder)
    builder.llm_component = MagicMock()
    builder.node_store_component = store  # type: ignore[assignment]
    return builder


@pytest.mark.parametrize("make_builder", [_semantic_builder, _tabular_builder])
async def test_tool_schema_artifacts_items_are_objects(
    store: FakeNodeStore, make_builder: Any
) -> None:
    spec = await make_builder(store).build_tool(context_filter=_ctx())
    artifacts = spec.input_schema["properties"]["artifacts"]
    array = next(s for s in artifacts["anyOf"] if s.get("type") == "array")
    items = array["items"]
    assert items["type"] == "object"
    assert set(items["required"]) == {"name", "artifact_id"}
    assert set(items["properties"]) == {"name", "artifact_id"}
    assert "artifact_id" in artifacts["description"]
    assert spec.input_schema["required"] == ["query"]
    # The schema must survive the round-trip into a llama-index tool
    spec.to_function_tool()


async def test_semantic_search_mismatch_never_builds_workflow(
    store: FakeNodeStore,
) -> None:
    builder = _semantic_builder(store)
    builder.build = AsyncMock()  # type: ignore[method-assign]
    spec = await builder.build_tool(context_filter=_ctx())
    with pytest.raises(ValueError, match=f"has artifact_id {REPORT_2024}"):
        await spec.async_fn(
            query="revenue",
            artifacts=[{"name": "2024-Annual-Report.md", "artifact_id": REPORT_2020}],
        )
    builder.build.assert_not_awaited()


async def test_semantic_search_resolved_ids_scope_the_workflow(
    store: FakeNodeStore,
) -> None:
    builder = _semantic_builder(store)
    workflow = MagicMock()
    workflow.run = AsyncMock(return_value=MagicMock(retrieval=MagicMock(nodes=[])))
    builder.build = AsyncMock(return_value=workflow)  # type: ignore[method-assign]
    builder.prompt_builder_service.create_context_prompt.return_value = (
        MagicMock(format=MagicMock(return_value="")),
        None,
    )
    spec = await builder.build_tool(context_filter=_ctx())
    await spec.async_fn(
        query="revenue",
        artifacts=[{"name": "2024-Annual-Report.md", "artifact_id": REPORT_2024}],
    )
    context_filter = builder.build.await_args.kwargs["context_filter"]
    assert context_filter.artifacts == [REPORT_2024]


@pytest.mark.parametrize("make_builder", [_semantic_builder, _tabular_builder])
async def test_empty_scope_intersection_never_widens(
    store: FakeNodeStore, make_builder: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty intersection must not become an unrestricted search."""
    builder = make_builder(store)
    builder.build = AsyncMock()
    spec = await builder.build_tool(context_filter=_ctx(artifacts=[REPORT_2024]))

    # Bypass the resolver to exercise the builder's own guard
    async def passthrough(refs: Any, **_: Any) -> list[str]:
        return [ref["artifact_id"] for ref in refs]

    module = __import__(type(builder).__module__, fromlist=["resolve_artifact_refs"])
    monkeypatch.setattr(module, "resolve_artifact_refs", passthrough)

    with pytest.raises(ValueError, match="none of them is available here"):
        await spec.async_fn(
            query="q",
            artifacts=[{"name": "2020-Annual-Report.md", "artifact_id": REPORT_2020}],
        )
    builder.build.assert_not_awaited()
