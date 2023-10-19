from pydantic import BaseModel, Field

from private_gpt.settings.settings_loader import load_active_profiles


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field("Port of PrivateGPT FastAPI server, defaults to 8001")


class DataSettings(BaseModel):
    local_data_folder: str = Field(
        description="Path to local storage."
        "It will be treated as an absolute path if it starts with /"
    )


class LLMSettings(BaseModel):
    mode: str = Field(enum=["local", "open_ai", "sagemaker", "mock"])


class LocalSettings(BaseModel):
    llm_hf_repo_id: str
    llm_hf_model_file: str
    embedding_hf_model_name: str


class SagemakerSettings(BaseModel):
    endpoint_name: str


class OpenAISettings(BaseModel):
    api_key: str


class UISettings(BaseModel):
    enabled: bool
    path: str


class Settings(BaseModel):
    server: ServerSettings
    data: DataSettings
    ui: UISettings
    llm: LLMSettings
    local: LocalSettings
    sagemaker: SagemakerSettings
    openai: OpenAISettings


settings = Settings(**load_active_profiles())
