"""Provide in-memory toolset repository implementation."""

from pydantic import Field

from private_gpt.components.toolsets.models.tool_set import ToolSet
from private_gpt.components.toolsets.repositories.toolset_repository import (
    ToolSetRepository,
)


class InMemoryToolSetRepository(ToolSetRepository):
    """Provide an in-memory toolset repository keyed by name."""

    by_name: dict[str, ToolSet] = Field(default_factory=dict)

    def save(self, toolset: ToolSet) -> ToolSet:
        """Save a toolset value in memory and return it."""
        self.by_name[toolset.name] = toolset
        return toolset

    def get_by_name(self, name: str) -> ToolSet | None:
        """Get a toolset by name from memory."""
        return self.by_name.get(name)

    def list(self) -> list[ToolSet]:
        """List all toolsets from memory."""
        return list(self.by_name.values())

    def delete(self, name: str) -> bool:
        """Delete one toolset by name from memory."""
        removed = self.by_name.pop(name, None)
        return removed is not None
