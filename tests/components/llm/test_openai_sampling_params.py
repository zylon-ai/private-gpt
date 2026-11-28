from types import SimpleNamespace
from typing import Any

import pytest

from private_gpt.components.llm.factories.completions.openai import (
    OpenAICompletionsFactory,
)
from private_gpt.components.llm.factories.responses.openai import (
    OpenAIResponsesFactory,
)
from private_gpt.settings.settings import LLMModelConfig, SamplingParams

FACTORIES: list[tuple[Any, str]] = [
    (OpenAICompletionsFactory, "max_tokens"),
    (OpenAIResponsesFactory, "max_output_tokens"),
]


def _settings() -> Any:
    return SimpleNamespace(
        openai=SimpleNamespace(
            api_base="https://api.openai.com/v1", api_key="test", request_timeout=None
        )
    )


def _llm(factory_cls: Any, params: SamplingParams) -> Any:
    config = LLMModelConfig(name="gpt-4o-mini", mode="openai", sampling_params=params)
    llm, _ = factory_cls(_settings())._create_llm(config)
    return llm


@pytest.mark.parametrize(("factory_cls", "max_tokens_field"), FACTORIES)
def test_configured_temperature_is_used(
    factory_cls: Any, max_tokens_field: str
) -> None:
    assert _llm(factory_cls, SamplingParams(temperature=0.7)).temperature == 0.7


@pytest.mark.parametrize(("factory_cls", "max_tokens_field"), FACTORIES)
def test_zero_temperature_is_kept(factory_cls: Any, max_tokens_field: str) -> None:
    assert _llm(factory_cls, SamplingParams(temperature=0.0)).temperature == 0.0


@pytest.mark.parametrize(("factory_cls", "max_tokens_field"), FACTORIES)
def test_configured_max_new_tokens_is_used(
    factory_cls: Any, max_tokens_field: str
) -> None:
    llm = _llm(factory_cls, SamplingParams(max_new_tokens=256))
    assert getattr(llm, max_tokens_field) == 256


@pytest.mark.parametrize(("factory_cls", "max_tokens_field"), FACTORIES)
def test_defaults_are_unchanged(factory_cls: Any, max_tokens_field: str) -> None:
    llm = _llm(factory_cls, SamplingParams())
    assert llm.temperature == 0.1
    assert getattr(llm, max_tokens_field) is None
