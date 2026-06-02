from enum import StrEnum


class LayerType(StrEnum):
    """Define supported context layer types.

    Layers are split into two categories:
    - Prompt layers  : rendered into the system prompt text sent to the LLM.
    - State layers   : carry structured data consumed programmatically by the
                       loop (tools, metadata) but not rendered as plain text.
    """

    # Prompt layers
    USER_INSTRUCTIONS = "USER_INSTRUCTIONS"
    RUNTIME_INSTRUCTIONS = "RUNTIME_INSTRUCTIONS"
    CONTEXT = "CONTEXT"

    SKILL_CATALOG = "SKILL_CATALOG"
    SKILL_BODY = "SKILL_BODY"
    TOOL_INSTRUCTIONS = "TOOL_INSTRUCTIONS"

    # State layers
    TOOL_DEFINITIONS = "TOOL_DEFINITIONS"
    DOCUMENT = "DOCUMENT"
