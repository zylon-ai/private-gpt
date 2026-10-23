"""Define toolset repository abstraction."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from private_gpt.components.toolsets.models.tool_set import ToolSet


class ToolSetRepository(BaseModel, ABC):
    """Define persistence operations for registered toolsets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def save(self, toolset: ToolSet) -> ToolSet:
        """Persist a toolset and return the stored value."""

    @abstractmethod
    def get_by_name(self, name: str) -> ToolSet | None:
        """Find a toolset by name."""

    @abstractmethod
    def list(self) -> list[ToolSet]:
        """Return all registered toolsets."""

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete a toolset by name."""
