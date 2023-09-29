from pydantic import BaseModel

from private_gpt.util.settings_loader import load_active_profiles


class ServerSettings(BaseModel):
    env_name: str
    port: int


class SagemakerSettings(BaseModel):
    enabled: bool
    endpoint_name: str


class Settings(BaseModel):
    server: ServerSettings
    sagemaker: SagemakerSettings


settings = Settings(**load_active_profiles())
