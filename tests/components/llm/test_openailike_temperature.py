from types import SimpleNamespace
from typing import Any

import pytest

from private_gpt.components.llm.factories.completions.openailike import (
    OpenAILikeCompletionsFactory,
)
from private_gpt.components.llm.factories.responses.openailike import (
    OpenAILikeResponsesFactory,
)
from private_gpt.settings.settings import LLMModelConfig, SamplingParams

FACTORIES: list[Any] = [OpenAILikeCompletionsFactory, OpenAILikeResponsesFactory]


def _settings() -> Any:
    return SimpleNamespace(
        openai=SimpleNamespace(
            api_base="http://localhost:1/v1", api_key="test", request_timeout=None
        )
    )


def _temperature(factory_cls: Any, params: SamplingParams) -> float:
    config = LLMModelConfig(name="mock", mode="openailike", sampling_params=params)
    llm, _ = factory_cls(_settings())._create_llm(config)
    return float(llm.temperature)


@pytest.mark.parametrize("factory_cls", FACTORIES)
def test_zero_temperature_is_kept(factory_cls: Any) -> None:
    assert _temperature(factory_cls, SamplingParams(temperature=0.0)) == 0.0


@pytest.mark.parametrize("factory_cls", FACTORIES)
def test_default_temperature_is_unchanged(factory_cls: Any) -> None:
    assert _temperature(factory_cls, SamplingParams()) == 0.1
