import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.workflows.others.tree_summarize_synthesizer import (
    TreeSummarizeSynthesizer,
)


@pytest.mark.asyncio
async def test_repack_runs_off_the_event_loop_thread() -> None:
    loop_thread = threading.get_ident()
    repack_threads: list[int] = []

    def repack(*_args: object, **kwargs: object) -> list[str]:
        repack_threads.append(threading.get_ident())
        return ["only chunk"]

    prompt_helper = MagicMock()
    prompt_helper.repack.side_effect = repack
    llm = MagicMock()
    # the synthesizer calls achat/acomplete directly (so llm_kwargs reach the LLM)
    llm.metadata.is_chat_model = False
    llm.acomplete = AsyncMock(return_value=MagicMock(text="summary"))

    synthesizer = TreeSummarizeSynthesizer(
        llm=llm, prompt_helper=prompt_helper, use_async=True
    )
    response = await synthesizer.aget_response("q", ["a", "b"])

    assert response == "summary"
    assert repack_threads
    assert loop_thread not in repack_threads
