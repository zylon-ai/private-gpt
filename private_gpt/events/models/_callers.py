from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DirectCaller(BaseModel):
    type: Literal["direct"] = Field()

    model_config = ConfigDict(extra="allow")


class ServerToolCaller(BaseModel):
    tool_id: str = Field(
        description="Server tool identifier.",
        pattern=r"^srvtoolu_[a-zA-Z0-9_]+$",
    )
    type: Literal["code_execution_20250825"] = Field(
        description="Caller type discriminator."
    )

    model_config = ConfigDict(extra="allow")


class ServerToolCaller20260120(BaseModel):
    tool_id: str = Field(
        description="Server tool identifier.",
        pattern=r"^srvtoolu_[a-zA-Z0-9_]+$",
    )
    type: Literal["code_execution_20260120"] = Field(
        description="Caller type discriminator."
    )

    model_config = ConfigDict(extra="allow")


ToolCaller = Annotated[
    DirectCaller | ServerToolCaller | ServerToolCaller20260120,
    Field(discriminator="type"),
]
