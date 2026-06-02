from types import SimpleNamespace

from private_gpt.server.models.models_service import ModelsService
from private_gpt.settings.settings import LLMModelConfig, SamplingParams


class _FakeLLMComponent:
    def __init__(
        self, models: dict[str, LLMModelConfig], default_model_id: str
    ) -> None:
        self.llm_models = models
        self.default_model_id = default_model_id

    def get_config(self, model_id: str | None) -> LLMModelConfig:
        if model_id is None or model_id not in self.llm_models:
            raise ValueError(f"Unknown model: {model_id}")
        return self.llm_models[model_id]


def _make_model(
    name: str,
    support_reasoning: bool | None,
    support_image: int | None,
) -> LLMModelConfig:
    return LLMModelConfig(
        name=name,
        mode="mock",
        support_reasoning=support_reasoning,
        support_image=support_image,
        sampling_params=SamplingParams(max_new_tokens=2048),
    )


def test_capabilities_do_not_fallback_to_default_model_config() -> None:
    default_model = _make_model(
        "default-model", support_reasoning=True, support_image=1
    )
    target_model = _make_model(
        "target-model", support_reasoning=None, support_image=None
    )
    llm_component = _FakeLLMComponent(
        models={
            default_model.name: default_model,
            target_model.name: target_model,
        },
        default_model_id=default_model.name,
    )
    settings = SimpleNamespace(
        chat=SimpleNamespace(allow_reasoning=True, allow_generate_citations=False),
        sandbox=SimpleNamespace(enabled=True),
    )
    service = ModelsService(llm_component=llm_component, settings=settings)

    model_info = service.get_model(target_model.name)

    assert model_info.capabilities is not None
    assert model_info.capabilities.thinking.supported is False
    assert model_info.capabilities.thinking.types.enabled.supported is False
    assert model_info.capabilities.thinking.types.adaptive.supported is False
    assert model_info.capabilities.image_input.supported is False
    assert model_info.capabilities.citations.supported is False
    assert model_info.capabilities.code_execution.supported is False
    assert model_info.capabilities.context_management.supported is False
    assert model_info.capabilities.context_management.clear_thinking_20251015 is None
    assert model_info.capabilities.context_management.clear_tool_uses_20250919 is None
    assert model_info.capabilities.context_management.compact_20260112 is None
    assert model_info.capabilities.effort.supported is False
    assert model_info.capabilities.effort.low.supported is False
    assert model_info.capabilities.effort.medium.supported is False
    assert model_info.capabilities.effort.high.supported is False
    assert model_info.capabilities.effort.max.supported is False


def test_global_reasoning_disable_overrides_model_support() -> None:
    default_model = _make_model(
        "default-model", support_reasoning=True, support_image=None
    )
    llm_component = _FakeLLMComponent(
        models={default_model.name: default_model},
        default_model_id=default_model.name,
    )
    settings = SimpleNamespace(
        chat=SimpleNamespace(allow_reasoning=False, allow_generate_citations=True),
        sandbox=SimpleNamespace(enabled=False),
    )
    service = ModelsService(llm_component=llm_component, settings=settings)

    model_info = service.get_model(default_model.name)

    assert model_info.capabilities is not None
    assert model_info.capabilities.thinking.supported is False
    assert model_info.capabilities.thinking.types.enabled.supported is False
    assert model_info.capabilities.thinking.types.adaptive.supported is False
