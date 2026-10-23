from datetime import UTC, datetime

from injector import inject, singleton

from private_gpt.chat.input_models import (
    CapabilitySupportOutput,
    ContextManagementCapabilityOutput,
    CountCapabilitySupportOutput,
    EffortCapabilityOutput,
    ModelCapabilitiesOutput,
    ModelInfoOutput,
    ModelListOutput,
    ThinkingCapabilityOutput,
    ThinkingTypesOutput,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.settings.settings import LLMModelConfig, Settings


@singleton
class ModelsService:
    @inject
    def __init__(self, llm_component: LLMComponent, settings: Settings) -> None:
        self.llm_component = llm_component
        self.settings = settings

    def resolve_model_config(self, model_id: str | None) -> LLMModelConfig:
        return self.llm_component.get_config(model_id)

    @staticmethod
    def _resolve_model_flag(
        model_value: bool | None, default_value: bool | None
    ) -> bool:
        if model_value is not None:
            return model_value
        if default_value is not None:
            return default_value
        return False

    @staticmethod
    def _resolve_model_support(
        model_value: int | None, default_value: int | None
    ) -> bool:
        if model_value is not None:
            return model_value > 0
        if default_value is not None:
            return default_value > 0
        return False

    def _build_capabilities(self, config: LLMModelConfig) -> ModelCapabilitiesOutput:
        supports_reasoning = bool(config.support_reasoning)
        supports_image_input = bool(config.support_image)
        supports_audio_input = bool(config.support_audio)

        thinking_enabled = self.settings.chat.allow_reasoning and supports_reasoning

        supported = CapabilitySupportOutput(supported=True)
        unsupported = CapabilitySupportOutput(supported=False)

        return ModelCapabilitiesOutput(
            batch=unsupported,
            citations=CapabilitySupportOutput(
                supported=self.settings.chat.allow_generate_citations
            ),
            thinking=ThinkingCapabilityOutput(
                supported=thinking_enabled,
                types=ThinkingTypesOutput(
                    adaptive=unsupported,
                    enabled=CapabilitySupportOutput(supported=thinking_enabled),
                ),
            ),
            # TODO: Change when we can configure the effort per model
            effort=EffortCapabilityOutput(
                supported=thinking_enabled,
                low=supported if thinking_enabled else unsupported,
                medium=supported if thinking_enabled else unsupported,
                high=supported if thinking_enabled else unsupported,
                max=supported if thinking_enabled else unsupported,
                xhigh=supported if thinking_enabled else unsupported,
            ),
            image_input=CountCapabilitySupportOutput(
                supported=supports_image_input, maximum=config.support_image or 0
            ),
            audio_input=CountCapabilitySupportOutput(
                supported=supports_audio_input, maximum=config.support_audio or 0
            ),
            pdf_input=unsupported,
            structured_outputs=supported,
            code_execution=CapabilitySupportOutput(
                # TODO: Enable when we have code execution
                supported=False
            ),
            context_management=ContextManagementCapabilityOutput(
                clear_thinking_20251015=None,
                clear_tool_uses_20250919=None,
                compact_20260112=None,
                supported=False,
            ),
        )

    def _to_model_info(self, config: LLMModelConfig) -> ModelInfoOutput:
        return ModelInfoOutput(
            id=config.name,
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            display_name=config.alias or config.name,
            type="model",
            max_tokens=config.sampling_params.max_new_tokens,
            max_input_tokens=config.context_window,
            capabilities=self._build_capabilities(config),
        )

    def list_models(
        self,
        before_id: str | None,
        after_id: str | None,
        limit: int,
    ) -> ModelListOutput:
        if before_id and after_id:
            raise ValueError("Only one of before_id or after_id can be provided")

        all_models = list(self.llm_component.llm_models.values())
        all_models.sort(key=lambda model: model.name)
        model_ids = [model.name for model in all_models]

        if after_id:
            if after_id not in model_ids:
                raise ValueError(f"Unknown model id: {after_id}")
            start_idx = model_ids.index(after_id) + 1
            available = all_models[start_idx:]
            page = available[:limit]
            has_more = len(available) > len(page)
        elif before_id:
            if before_id not in model_ids:
                raise ValueError(f"Unknown model id: {before_id}")
            end_idx = model_ids.index(before_id)
            available = all_models[:end_idx]
            page = available[max(0, len(available) - limit) :]
            has_more = len(available) > len(page)
        else:
            available = all_models
            page = available[:limit]
            has_more = len(available) > len(page)

        data = [self._to_model_info(model) for model in page]
        return ModelListOutput(
            data=data,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
            has_more=has_more,
        )

    def get_model(self, model_id: str) -> ModelInfoOutput:
        return self._to_model_info(self.resolve_model_config(model_id))
