from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    token_endpoint: str | None = Field(
        default=None,
        description="The OAuth token endpoint used to refresh the authorization token.",
    )
    oauth_resource: str | None = Field(
        default=None,
        description="The OAuth resource associated with the MCP server.",
    )
    artifact_id: str | None = Field(
        default=None,
        description="The Zylon artifact ID associated with this MCP server.",
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
            missing = [
                name
                for name in ("client_id", "token_endpoint")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"refresh_token requires {', '.join(missing)}")
        return self
