from pydantic import BaseModel

from private_gpt.util.settings_loader import load_active_profiles


class ServerSettings(BaseModel):
    env_name: str
    port: int


class LLMSettings(BaseModel):
    default_llm: str


class LocalLLMSettings(BaseModel):
    enabled: bool
    model_file: str


class SagemakerSettings(BaseModel):
    enabled: bool
    endpoint_name: str


class OpenAISettings(BaseModel):
    enabled: bool
    api_key: str


class UISettings(BaseModel):
    enabled: bool
    path: str


class Settings(BaseModel):
    server: ServerSettings
    ui: UISettings
    llm: LLMSettings
    local_llm: LocalLLMSettings
    sagemaker: SagemakerSettings
    openai: OpenAISettings


settings = Settings(**load_active_profiles())
