from llama_index.core.agent import ToolCallResult
from llama_index.core.agent.workflow import (
    AgentStream as LIAgentStream,
)
from llama_index.core.tools import BaseTool
from workflows.events import Event

from private_gpt.events.models import BasicContentBlockType


class AgentStream(LIAgentStream):
    """Agent stream."""

    reasoning: str | None = None
    tldr: list[BasicContentBlockType | None] | None = None


class ToolResultOrNothing(Event):
    tool: BaseTool
    result: ToolCallResult | None = None
    executed: bool = False
