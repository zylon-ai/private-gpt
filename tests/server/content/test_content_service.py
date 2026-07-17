import asyncio
import threading

import pytest

from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode
from private_gpt.server.content.content_router import ContentTree
from private_gpt.server.content.content_service import (
    ContentRequestLimitError,
    ContentService,
)


@pytest.mark.asyncio
async def test_document_retrieval_advances_generator_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(ContentService)
    worker_threads: list[int] = []

    def retrieve(*args: object, **kwargs: object):
        del args, kwargs
        worker_threads.append(threading.get_ident())
        yield "artifact", TextNode(text="content")

    monkeypatch.setattr(service, "_retrieve_document_node", retrieve)
    main_thread = threading.get_ident()

    iterator = await service.retrieve_document_nodes_async(context_filter=object())  # type: ignore[arg-type]
    artifact, node = await anext(iterator)

    assert artifact == "artifact"
    assert node.get_content() == "content"
    assert worker_threads == [worker_threads[0]]
    assert worker_threads[0] != main_thread


@pytest.mark.asyncio
async def test_document_retrieval_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(ContentService)
    started = threading.Event()
    release = threading.Event()

    def retrieve(*args: object, **kwargs: object):
        del args, kwargs
        started.set()
        release.wait(timeout=1)
        yield "artifact", TextNode(text="content")

    monkeypatch.setattr(service, "_retrieve_document_node", retrieve)
    iterator = await service.retrieve_document_nodes_async(context_filter=object())  # type: ignore[arg-type]
    pending = asyncio.create_task(anext(iterator))

    await asyncio.to_thread(started.wait, 1)
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    await pending


def test_filter_tree_nodes_enforces_node_limit() -> None:
    service = object.__new__(ContentService)
    service.max_content_nodes = 2
    root = TextNode(text="root")
    root.children = [TextNode(text="one"), TextNode(text="two")]

    with pytest.raises(ContentRequestLimitError, match="node limit"):
        list(service._filter_tree_nodes(root))


def test_content_tree_conversion_handles_deep_trees() -> None:
    root = TextNode(text="root")
    current = root
    for index in range(1_500):
        child = TextNode(text=str(index))
        current.children = [child]
        current = child

    converted = ContentTree.from_node(root, TreeMetadataMode.NONE)

    assert converted.content == "root"
    assert len(list(root.flatten())) == 1_501
