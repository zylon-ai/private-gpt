def escape_markdown_cell(value: str) -> str:
    """Keep a cell from breaking the Markdown table it lands in."""
    return (
        value.replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "")
    )


def pad_or_trim_row(row: list[str], column_count: int) -> list[str]:
    if len(row) < column_count:
        return row + [""] * (column_count - len(row))
    if len(row) > column_count:
        return row[:column_count]
    return row


def render_markdown_table(rows: list[list[str]]) -> str:
    header, *body = rows
    column_count = len(header)

    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    for row in body:
        row = pad_or_trim_row(row, column_count)
        lines.append(
            "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |"
        )

    return "\n".join(lines)
