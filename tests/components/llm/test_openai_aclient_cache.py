import asyncio

from private_gpt.components.llm.custom.openailike import PatchedOpenAILikeLLM


def _llm() -> PatchedOpenAILikeLLM:
    return PatchedOpenAILikeLLM(
        model="mock",
        api_base="http://localhost:1/v1",
        api_key="test",
        reuse_client=False,
        is_chat_model=True,
    )


def test_one_async_client_per_running_loop() -> None:
    llm = _llm()

    async def two_clients() -> tuple[object, object]:
        return llm._get_aclient(), llm._get_aclient()

    first_a, first_b = asyncio.run(two_clients())
    second_a, _ = asyncio.run(two_clients())

    assert first_a is first_b
    assert second_a is not first_a


def test_fresh_async_client_outside_a_loop() -> None:
    llm = _llm()
    assert llm._get_aclient() is not llm._get_aclient()
