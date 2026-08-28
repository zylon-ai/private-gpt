from private_gpt.components.skills.parser import parse_skill_markdown


def test_parses_unquoted_multiline_description_with_colons() -> None:
    skill = """---
name: blog-post-generator
description: How to turn notes into a polished draft. Use this skill whenever the user provides raw notes: from Notion exports or similar. The output should be dry, specific, and structured with H2 headers.
---

# Blog Post Generator
"""

    parsed = parse_skill_markdown(skill)

    assert parsed.frontmatter.name == "blog-post-generator"
    assert parsed.frontmatter.description.startswith("How to turn notes")
    assert "raw notes: from Notion exports" in parsed.frontmatter.description


def test_parses_unquoted_string_fields_with_colons_and_continuation_lines() -> None:
    skill = """---
name: example-skill
description: Use this skill: when needed
license: MIT: License
compatibility: Works with: Claude
allowed-tools: Read: Write
  and Edit
---

Body
"""

    parsed = parse_skill_markdown(skill)

    assert parsed.frontmatter.license == "MIT: License"
    assert parsed.frontmatter.compatibility == "Works with: Claude"
    assert parsed.frontmatter.allowed_tools == ["Read:", "Write", "and", "Edit"]


def test_parses_unquoted_description_continuation_lines() -> None:
    skill = """---
name: example-skill
description: First line of the description
  and its continuation on another line.
---

Body
"""

    parsed = parse_skill_markdown(skill)

    assert parsed.frontmatter.description == (
        "First line of the description and its continuation on another line."
    )
