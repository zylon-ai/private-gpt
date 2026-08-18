from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class McpServerToolConfig(BaseModel):
    """Configuration for tool filtering from the MCP server."""

    enabled: bool = Field(
        default=True,
        description="Enable tool filtering for the MCP server.",
    )

    allowed_tools: list[str] | None = Field(
        default=None,
        description="List of enabled tools to filter from the MCP server.",
    )


class McpServerConfig(BaseModel):
    """Configuration for the MCP server."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(
        default="mcp",
        description="A name for the MCP server configuration.",
    )
    url: str = Field(
        description="The URL of the MCP server to connect to.",
    )
    authorization_token: str | None = Field(
        default=None,
        description="The authorization token to use when connecting to the MCP server.",
    )
    refresh_token: str | None = Field(
        default=None,
        description="The OAuth refresh token used to renew the authorization token.",
    )
    client_id: str | None = Field(
        default=None,
        description="The OAuth client ID associated with the refresh token.",
    )
    client_secret: str | None = Field(
        default=None,
        description="The optional OAuth client secret associated with the client ID.",
    )
    token_endpoint_auth_method: (
        Literal["none", "client_secret_basic", "client_secret_post"] | None
    ) = Field(
        default=None,
        description=(
            "The authentication method registered for the OAuth token endpoint. "
            "Defaults to client_secret_basic when a client secret is provided."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("_meta", "metadata"),
        serialization_alias="_meta",
        description="Opaque metadata copied into internal MCP events.",
    )
    tool_configuration: McpServerToolConfig = Field(
        default_factory=McpServerToolConfig,
        description="Configuration for tool filtering from the MCP server",
    )

    @model_validator(mode="before")
    @classmethod
    def strip_string_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.strip() if v and isinstance(v, str) else v for k, v in values.items()
        }

    @model_validator(mode="after")
    def validate_refresh_token_config(self) -> "McpServerConfig":
        if self.refresh_token:
            missing = [name for name in ("client_id",) if not getattr(self, name)]
            if missing:
                raise ValueError(f"refresh_token requires {', '.join(missing)}")
        return self
