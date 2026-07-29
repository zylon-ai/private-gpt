"""Render-time deduplication of ``ContextStack.to_system_prompt``.

Layers stay isolated in the stack; deduplication happens only at render. The
contract enforced here:

- The same text never reaches the LLM twice (stale duplicate drop).
- A re-ingested ``UserInstructionsLayer`` that aggregates a previous response
  (header + guidelines) is discarded once the freshly-generated isolated
  layers that the interceptors rebuilt for this iteration already reproduce
  its parts (snowballed-aggregate drop).
- The freshly-generated isolated layers, including the ``ContextPromptLayer``
  rebuilt with the latest documents accumulated across iterations, must
  survive over any stale aggregate that embeds an older version of them.
"""


from private_gpt.components.context.models.context_layer import (
    ContextPromptLayer,
    RuntimeInstructionsLayer,
    SkillBodyLayer,
    ToolInstructionsLayer,
    UserInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack


def _render(stack: ContextStack) -> list[str]:
    return [b.text for b in stack.to_system_prompt() if b.text]


class TestRenderTimeDeduplication:
    def test_identical_layers_collapse_to_one_block(self) -> None:
        text = "You are Zylon, an AI assistant.\nCurrent date: 2026-07-28."
        stack = ContextStack(
            layers=[
                UserInstructionsLayer(text=text, source="request"),
                RuntimeInstructionsLayer(text=text, source="platform_header"),
            ]
        )

        rendered = _render(stack)

        assert rendered == [text]

    def test_stale_duplicate_lower_priority_is_dropped(self) -> None:
        """Runtime header is kept; aggregate UserInstructions is dropped."""
        header = "You are Zylon, an AI assistant."
        guideline = "<response_formatting>\nWrite clearly.\n</response_formatting>"
        bloated = f"{header}\n\n{guideline}"
        stack = ContextStack(
            layers=[
                UserInstructionsLayer(text=bloated, source="request"),
                RuntimeInstructionsLayer(text=header, source="platform_header"),
                ToolInstructionsLayer(
                    tool_name="response_formatting",
                    instructions=guideline,
                    source="platform:tool_instructions",
                ),
            ]
        )

        rendered = _render(stack)
        assert rendered.count(header) == 1
        assert rendered.count(guideline) == 1
        # The bloated aggregate must NOT survive: only the isolated layers do
        assert bloated not in rendered
        assert header in rendered
        assert guideline in rendered

    def test_latest_context_prompt_survives_stale_aggregate(self) -> None:
        """A re-ingested UserInstructions embedding an *older* rendered
        context prompt is discarded; the fresh ContextPromptLayer (rebuilt
        from the latest documents accumulated across iterations) survives.
        """
        stale_ctx = "<context_doc ids=[LVSE]>\nold content\n</context_doc>"
        fresh_ctx = "<context_doc ids=[LVSE, WR2J]>\nlatest content\n</context_doc>"
        # Aggregate layer reproduces the *stale* version of the context.
        bloated = f"You are Zylon.\n\n{stale_ctx}\n\n<response_formatting>...</response_formatting>"
        stack = ContextStack(
            layers=[
                UserInstructionsLayer(text=bloated, source="request"),
                RuntimeInstructionsLayer(text="You are Zylon.", source="platform_header"),
                ContextPromptLayer(text=fresh_ctx, source="system_prompt"),
            ]
        )

        rendered = _render(stack)
        assert fresh_ctx in rendered, "Latest context prompt must survive"
        assert stale_ctx not in rendered, "Stale context prompt must be dropped"
        assert bloated not in rendered, "Snowballed aggregate must be dropped"

    def test_distinct_isolated_layers_kept(self) -> None:
        """No false positives: layers with non-overlapping content are kept."""
        header = "You are Zylon."
        guideline = "<response_formatting>\nWrite clearly.\n</response_formatting>"
        # SkillBodyLayer wraps the instructions in <skill_content name="...">
        skill_body = '<skill_content name="x">\nbody\n</skill_content>'
        stack = ContextStack(
            layers=[
                RuntimeInstructionsLayer(text=header, source="platform_header"),
                ToolInstructionsLayer(
                    tool_name="response_formatting",
                    instructions=guideline,
                    source="platform:tool_instructions",
                ),
                SkillBodyLayer(
                    skill_id="x",
                    name="x",
                    version="1",
                    instructions="body",
                    source="skill:x",
                ),
            ]
        )

        rendered = _render(stack)

        assert set(rendered) == {header, guideline, skill_body}

    def test_render_is_idempotent_across_iterations(self) -> None:
        """Simulate two iterations where the same starter stack is rendered
        twice — output must not grow with repeated calls.
        """
        header = "You are Zylon."
        stack = ContextStack(
            layers=[
                UserInstructionsLayer(text=f"{header}\n<old>", source="request"),
                RuntimeInstructionsLayer(text=header, source="platform_header"),
            ]
        )

        first = _render(stack)
        second = _render(ContextStack(layers=list(stack.layers)))

        assert first == second
        assert first.count(header) == 1
