from typing import Literal

from pydantic import BaseModel, Field

from private_gpt.settings.settings_loader import load_active_profiles


class CorsSettings(BaseModel):
    """CORS configuration.

    For more details on the CORS configuration, see:
    # * https://fastapi.tiangolo.com/tutorial/cors/
    # * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    """

    enabled: bool = Field(
        description="Flag indicating if CORS headers are set or not."
        "If set to True, the CORS headers will be set to allow all origins, methods and headers.",
        default=False,
    )
    allow_credentials: bool = Field(
        description="Indicate that cookies should be supported for cross-origin requests",
        default=False,
    )
    allow_origins: list[str] = Field(
        description="A list of origins that should be permitted to make cross-origin requests.",
        default=[],
    )
    allow_origin_regex: list[str] = Field(
        description="A regex string to match against origins that should be permitted to make cross-origin requests.",
        default=None,
    )
    allow_methods: list[str] = Field(
        description="A list of HTTP methods that should be allowed for cross-origin requests.",
        default=[
            "GET",
        ],
    )
    allow_headers: list[str] = Field(
        description="A list of HTTP request headers that should be supported for cross-origin requests.",
        default=[],
    )


class AuthSettings(BaseModel):
    """Authentication configuration.

    The implementation of the authentication strategy must
    """

    enabled: bool = Field(
        description="Flag indicating if authentication is enabled or not.",
        default=False,
    )
    secret: str = Field(
        description="The secret to be used for authentication. "
        "It can be any non-blank string. For HTTP basic authentication, "
        "this value should be the whole 'Authorization' header that is expected"
    )


class ServerSettings(BaseModel):
    env_name: str = Field(
        description="Name of the environment (prod, staging, local...)"
    )
    port: int = Field(description="Port of PrivateGPT FastAPI server, defaults to 8001")
    cors: CorsSettings = Field(
        description="CORS configuration", default=CorsSettings(enabled=False)
    )
    auth: AuthSettings = Field(
        description="Authentication configuration",
        default_factory=lambda: AuthSettings(enabled=False, secret="secret-key"),
    )


class DataSettings(BaseModel):
    local_data_folder: str = Field(
        description="Path to local storage."
        "It will be treated as an absolute path if it starts with /"
    )


class LLMSettings(BaseModel):
    mode: Literal["local", "open_ai", "sagemaker", "mock"]


class LocalSettings(BaseModel):
    llm_hf_repo_id: str
    llm_hf_model_file: str
    embedding_hf_model_name: str


class SagemakerSettings(BaseModel):
    llm_endpoint_name: str
    embedding_endpoint_name: str


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
