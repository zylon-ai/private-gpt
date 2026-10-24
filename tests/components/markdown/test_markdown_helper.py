from private_gpt.components.markdown.markdown_helper import MarkdownHelper


def test_emphasis_markers_are_stripped_from_prose() -> None:
    markdown = "Some **bold**, *italic* and _also italic_ text.\n"

    assert (
        MarkdownHelper.sanitize_markdown(markdown)
        == "Some bold, italic and also italic text.\n"
    )


def test_fenced_code_block_is_kept_verbatim() -> None:
    markdown = (
        "Example:\n"
        "```python\n"
        "def load_user_data(user_id):\n"
        "    return a * b * c, __name__\n"
        "```\n"
        "~~~\n"
        "snake_case_name = **kwargs\n"
        "~~~\n"
        "After **the** code.\n"
    )

    assert MarkdownHelper.sanitize_markdown(markdown) == (
        "Example:\n"
        "```python\n"
        "def load_user_data(user_id):\n"
        "    return a * b * c, __name__\n"
        "```\n"
        "~~~\n"
        "snake_case_name = **kwargs\n"
        "~~~\n"
        "After the code.\n"
    )


def test_fenced_code_block_spanning_batches_is_kept_verbatim() -> None:
    markdown = "```\nfoo_bar_baz\nx * y * z\n```\n*done*\n"

    assert (
        MarkdownHelper.sanitize_markdown(markdown, chunk_size=1, max_workers=4)
        == "```\nfoo_bar_baz\nx * y * z\n```\ndone\n"
    )


def test_inline_code_is_kept_verbatim() -> None:
    markdown = "Set `MAX_NEW_TOKENS` and call `__init__` or `a * b * c` **now**.\n"

    assert MarkdownHelper.sanitize_markdown(markdown) == (
        "Set `MAX_NEW_TOKENS` and call `__init__` or `a * b * c` now.\n"
    )


def test_intraword_underscores_are_kept() -> None:
    markdown = "Edit settings_local.yaml and set MAX_NEW_TOKENS for a _long_ answer.\n"

    assert MarkdownHelper.sanitize_markdown(markdown) == (
        "Edit settings_local.yaml and set MAX_NEW_TOKENS for a long answer.\n"
    )


def test_inline_code_inside_link_is_restored() -> None:
    markdown = "See [`load_user_data`](https://example.com/load_user_data) **here**.\n"

    assert MarkdownHelper.sanitize_markdown(markdown) == (
        "See [`load_user_data`](https://example.com/load_user_data) here.\n"
    )
