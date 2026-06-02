import pandas as pd


def df_to_minimal_markdown(df: pd.DataFrame, allow_empty: bool = True) -> str:
    # Pre-check
    if df.columns.empty:
        return ""
    if df.empty and not allow_empty:
        return ""

    # Convert all data to strings
    df_str = df.astype(str)

    # Function to format a row with one space before and after each cell's content
    def format_row(row: list[str]) -> str:
        result = "| " + " | ".join(str(cell).strip() for cell in row) + " |"
        return " ".join(result.split())

    # Create the header row and separator row
    header_row = format_row(list(df.columns))
    separator_row = "| " + " | ".join("-" for _ in df.columns) + " |"

    # Create the data rows
    data_rows = [format_row(row) for row in df_str.values.tolist()]

    # Combine all parts into the final Markdown table
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"
