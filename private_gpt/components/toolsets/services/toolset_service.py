"""Implement toolset management service."""

from pydantic import BaseModel, ConfigDict

from private_gpt.components.toolsets.errors import InvalidToolSetError
from private_gpt.components.toolsets.models.tool_set import ToolSet
from private_gpt.components.toolsets.repositories.toolset_repository import (
    ToolSetRepository,
)


class ToolSetService(BaseModel):
    """Manage registration and retrieval of named toolsets."""

    repository: ToolSetRepository

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def register(self, toolset: ToolSet) -> ToolSet:
        """Register one toolset after validating tool name uniqueness."""
        names = [tool.name for tool in toolset.tools]
        if len(names) != len(set(names)):
            raise InvalidToolSetError(
                f"ToolSet '{toolset.name}' has duplicate tool names"
            )
        return self.repository.save(toolset)

    def get(self, name: str) -> ToolSet | None:
        """Return one toolset by name when it exists."""
        return self.repository.get_by_name(name)

    def list(self) -> list[ToolSet]:
        """Return all registered toolsets."""
        return self.repository.list()

    def delete(self, name: str) -> bool:
        """Delete one toolset by name."""
        return self.repository.delete(name)
