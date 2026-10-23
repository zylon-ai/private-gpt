from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from private_gpt.components.skills.errors import SkillErrorCode, SkillValidationErrors
from private_gpt.components.skills.models.skill_entities import (
    SkillFrontmatter as EntitySkillFrontmatter,
)
from private_gpt.components.skills.parser import SkillFrontmatter, parse_skill_markdown

_SKILL_FIXTURES = Path(__file__).parents[2] / "fixtures" / "skills"


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


@pytest.mark.parametrize(
    ("filename", "code"),
    [
        ("metadata-empty-key.md", SkillErrorCode.METADATA_EMPTY_KEY),
        ("metadata-invalid-value.md", SkillErrorCode.METADATA_INVALID_VALUE),
    ],
)
def test_rejects_invalid_metadata_fixtures(filename: str, code: SkillErrorCode) -> None:
    skill = (_SKILL_FIXTURES / filename).read_text()

    with pytest.raises(SkillValidationErrors) as exc_info:
        parse_skill_markdown(skill)

    assert [error.code for error in exc_info.value.errors] == [code]


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        pytest.param({"": "value"}, SkillErrorCode.METADATA_EMPTY_KEY, id="empty-key"),
        pytest.param(
            {"  ": "value"}, SkillErrorCode.METADATA_EMPTY_KEY, id="blank-key"
        ),
        pytest.param(
            {0: "value"}, SkillErrorCode.METADATA_INVALID_VALUE, id="numeric-key"
        ),
        pytest.param(
            {False: "value"}, SkillErrorCode.METADATA_INVALID_VALUE, id="boolean-key"
        ),
        pytest.param(
            {None: "value"}, SkillErrorCode.METADATA_INVALID_VALUE, id="null-key"
        ),
        pytest.param(
            {"test-value": {"nested": "value"}},
            SkillErrorCode.METADATA_INVALID_VALUE,
            id="mapping-value",
        ),
        pytest.param(
            {"test-value": ["value"]},
            SkillErrorCode.METADATA_INVALID_VALUE,
            id="list-value",
        ),
        pytest.param(
            {"test-value": {"value"}},
            SkillErrorCode.METADATA_INVALID_VALUE,
            id="set-value",
        ),
        pytest.param(
            {"test-value": None}, SkillErrorCode.METADATA_INVALID_VALUE, id="null-value"
        ),
        pytest.param([], SkillErrorCode.METADATA_INVALID_VALUE, id="list-metadata"),
        pytest.param(
            "value", SkillErrorCode.METADATA_INVALID_VALUE, id="string-metadata"
        ),
        pytest.param(42, SkillErrorCode.METADATA_INVALID_VALUE, id="numeric-metadata"),
        pytest.param(
            False, SkillErrorCode.METADATA_INVALID_VALUE, id="boolean-metadata"
        ),
    ],
)
def test_rejects_invalid_metadata(metadata: object, code: SkillErrorCode) -> None:
    frontmatter = {
        "name": "example-skill",
        "description": "A skill",
        "metadata": metadata,
    }
    skill = f"---\n{yaml.safe_dump(frontmatter)}---\nBody"

    with pytest.raises(SkillValidationErrors) as exc_info:
        parse_skill_markdown(skill)

    assert [error.code for error in exc_info.value.errors] == [code]
    with pytest.raises(ValidationError, match="metadata"):
        EntitySkillFrontmatter.model_validate(frontmatter)


@pytest.mark.parametrize("model", [SkillFrontmatter, EntitySkillFrontmatter])
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (None, None),
        ({}, {}),
        (
            {"author": "platform-team", "empty": ""},
            {"author": "platform-team", "empty": ""},
        ),
        (
            {"version": 1.2, "count": 0, "enabled": True, "disabled": False},
            {"version": "1.2", "count": "0", "enabled": "True", "disabled": "False"},
        ),
    ],
)
def test_preserves_valid_metadata(
    model: type[SkillFrontmatter] | type[EntitySkillFrontmatter],
    metadata: object,
    expected: dict[str, str] | None,
) -> None:
    frontmatter = model.model_validate(
        {"name": "example-skill", "description": "A skill", "metadata": metadata}
    )

    assert frontmatter.metadata == expected


def test_metadata_errors_are_aggregated_with_other_frontmatter_errors() -> None:
    skill = """---
name: INVALID
metadata:
  test-value:
    nested: value
---
Body
"""

    with pytest.raises(SkillValidationErrors) as exc_info:
        parse_skill_markdown(skill)

    assert {error.code for error in exc_info.value.errors} == {
        SkillErrorCode.NAME_INVALID_FORMAT,
        SkillErrorCode.DESCRIPTION_REQUIRED,
        SkillErrorCode.METADATA_INVALID_VALUE,
    }
