import re

import pytest

from private_gpt.components.tabular.database_query_generator import (
    DatabaseQueryGenerator,
)

SUPPORTED_CONNECTION_STRINGS = [
    "postgresql://user:pass@localhost:5432/testdb",
    "mssql+pyodbc://user:pass@server:1433/testdb?driver=ODBC+Driver+18+for+SQL+Server",
    "mysql+mysqldb://user:pass@localhost:3306/testdb?charset=utf8mb4",
]


@pytest.mark.parametrize(
    ("connection_string", "expected_connection_string"),
    [
        (
            "mysql://user:pass@localhost:3306/testdb",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
        ),
        (
            "mysql+mysqldb://user:pass@localhost:3306/testdb",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
        ),
        (
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=latin1",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=latin1",
        ),
    ],
)
def test_mysql_connection_string_is_normalized_to_pymysql(
    connection_string: str,
    expected_connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    assert generator.connection_string == expected_connection_string


def compare_sql(sql1: str, sql2: str, insensitive: bool = True) -> None:
    # Remove spaces and newlines
    sql1 = re.sub(r"\s+", " ", sql1).strip()
    sql2 = re.sub(r"\s+", " ", sql2).strip()

    # Lowercase for case-insensitive comparison
    if insensitive:
        sql1 = sql1.lower()
        sql2 = sql2.lower()

    # Handle SQLGlot normalizations
    if insensitive:
        # SQLGlot may add/remove AS keywords
        sql1 = re.sub(r"\bas\b", "", sql1).strip()
        sql2 = re.sub(r"\bas\b", "", sql2).strip()

        # Normalize multiple spaces again after AS removal
        sql1 = re.sub(r"\s+", " ", sql1)
        sql2 = re.sub(r"\s+", " ", sql2)

    assert sql1 == sql2, f"Expected: '{sql2}'\nActual: '{sql1}'"


def get_expected_result(input_sql: str, connection_string: str) -> str:
    result = input_sql

    if input_sql.endswith(";"):
        result = input_sql[:-1].strip()

    if "postgresql" in connection_string or "mysql" in connection_string:
        # Convert SQL Server TOP to LIMIT
        top_match = re.search(r"TOP\s+(\d+)", result, re.IGNORECASE)
        if top_match:
            number = top_match.group(1)
            # Remove TOP n from the SQL
            result = re.sub(r"TOP\s+\d+", "", result, flags=re.IGNORECASE)
            # Clean up extra spaces
            result = re.sub(r"\s+", " ", result).strip()
            # Add LIMIT at the end
            result = f"{result} LIMIT {number}"

    elif "mssql" in connection_string:
        # SQL Server specific transformations
        # COUNT(*) becomes COUNT_BIG(*) in SQL Server
        result = re.sub(r"\bCOUNT\s*\(", "COUNT_BIG(", result, flags=re.IGNORECASE)

    # SQLGlot adds AS keywords and normalizes to lowercase
    result = result.lower()

    # Add AS keywords where SQLGlot would add them
    # Pattern: "FROM table alias" becomes "FROM table AS alias"
    result = re.sub(r"(\bfrom\s+\w+(?:\.\w+)?)\s+([a-zA-Z_]\w*)\b", r"\1 as \2", result)

    return result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_remove_red_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_red: str = "\x1b[31mSELECT * FROM users\x1b[0m"

    result: str = generator._extract_sql_code(sql_with_red)
    expected: str = "SELECT * FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_remove_bold_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_bold: str = "\x1b[1mSELECT\x1b[0m * FROM \x1b[1musers\x1b[0m"

    result: str = generator._extract_sql_code(sql_with_bold)
    expected: str = "SELECT * FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_remove_underline_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_underline: str = "SELECT \x1b[4mTOP 5\x1b[0m\nFROM users"

    result: str = generator._extract_sql_code(sql_with_underline)
    expected: str = "SELECT TOP 5\nFROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_remove_multiple_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    complex_sql: str = (
        "\x1b[31m\x1b[1mSELECT\x1b[0m \x1b[4mTOP 5\x1b[0m\n"
        "\x1b[32mc.CityID\x1b[0m,\n"
        "\x1b[33mc.CityName\x1b[0m\n"
        "FROM \x1b[36mApplication.Cities\x1b[0m c"
    )

    result: str = generator._extract_sql_code(complex_sql)
    expected: str = (
        "SELECT TOP 5\n" "c.CityID,\n" "c.CityName\n" "FROM Application.Cities c"
    )
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_sql_without_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    clean_sql: str = "SELECT * FROM users WHERE id = 1"

    result: str = generator._extract_sql_code(clean_sql)
    expected: str = "SELECT * FROM users WHERE id = 1"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_ansi_codes_in_markdown_blocks(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    markdown_with_ansi: str = """
    ```sql
    \x1b[31mSELECT\x1b[0m *
    FROM \x1b[32musers\x1b[0m
    ```
    """

    result: str = generator._extract_sql_code(markdown_with_ansi)
    expected: str = "SELECT * \nFROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_complex_ansi_sequence(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_bg: str = "\x1b[41m\x1b[37mSELECT\x1b[0m * FROM users"

    result: str = generator._extract_sql_code(sql_with_bg)
    expected: str = "SELECT * FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_ansi_codes_at_line_boundaries(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_multiline: str = (
        "\x1b[31mSELECT *\x1b[0m\n"
        "\x1b[32mFROM users\x1b[0m\n"
        "\x1b[33mWHERE active = 1\x1b[0m"
    )

    result: str = generator._extract_sql_code(sql_multiline)
    expected: str = "SELECT *\n" "FROM users\n" "WHERE active = 1"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_ansi_codes_in_sql_strings(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_string_ansi: str = "SELECT '\x1b[31mHello\x1b[0m' as greeting FROM users"

    result: str = generator._extract_sql_code(sql_with_string_ansi)
    expected: str = "SELECT 'Hello' as greeting FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_empty_string_with_ansi(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    only_ansi: str = "\x1b[31m\x1b[0m"

    result: str = generator._extract_sql_code(only_ansi)

    assert result == ""
    assert "\x1b" not in result


@pytest.mark.parametrize(
    ("input_connection_string", "expected_connection_string"),
    [
        (
            "mysql://user:pass@localhost:3306/testdb",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
        ),
        (
            "mysql+mysqldb://user:pass@localhost:3306/testdb",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
        ),
        (
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
            "mysql+pymysql://user:pass@localhost:3306/testdb?charset=utf8mb4",
        ),
    ],
)
def test_mysql_connection_string_normalizes_to_pymysql(
    input_connection_string: str,
    expected_connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(input_connection_string)

    assert generator.connection_string == expected_connection_string


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_whitespace_preservation(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    sql_with_spaces: str = (
        "SELECT    \x1b[31mCOUNT(*)\x1b[0m   AS total\n"
        "FROM       users\n"
        "WHERE      \x1b[32mactive = 1\x1b[0m"
    )

    result: str = generator._extract_sql_code(sql_with_spaces)
    expected: str = (
        "SELECT    COUNT(*)   AS total\n" "FROM       users\n" "WHERE      active = 1"
    )
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_large_sql_with_many_ansi_codes(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    large_sql_parts: list[str] = ["\x1b[3mSELECT"]
    for i in range(10):  # Reduced for exact testing
        large_sql_parts.append(f"field_{i}\x1b[0m,")
    large_sql_parts[-1] = large_sql_parts[-1].rstrip(",")  # Remove last comma
    large_sql: str = "\n".join(large_sql_parts) + "\nFROM big_table"

    result: str = generator._extract_sql_code(large_sql)

    expected_parts: list[str] = ["SELECT"]
    for i in range(10):
        expected_parts.append(f"field_{i},")
    expected_parts[-1] = expected_parts[-1].rstrip(",")  # Remove last comma
    expected: str = "\n".join(expected_parts) + "\nFROM big_table"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_markdown_extraction_with_sql_keyword(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    markdown_sql: str = """
    ```sql
    \x1b[32mSELECT\x1b[0m id, name
    FROM users
    ```
    """

    result: str = generator._extract_sql_code(markdown_sql)
    expected: str = "SELECT id, name \nFROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_markdown_extraction_without_sql_keyword(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    markdown_no_sql: str = """
    ```
    \x1b[31mSELECT\x1b[0m * FROM products
    ```
    """

    result: str = generator._extract_sql_code(markdown_no_sql)
    expected: str = "SELECT * FROM products"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_no_markdown_blocks(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    plain_sql: str = "\x1b[33mSELECT\x1b[0m count(*) FROM orders"

    result: str = generator._extract_sql_code(plain_sql)
    expected: str = "SELECT count(*) FROM orders"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_opening_and_end_markdown(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    partial_markdown: str = "```sql\n\x1b[31mSELECT\x1b[0m * FROM users```"

    result: str = generator._extract_sql_code(partial_markdown)
    expected: str = "SELECT * FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


@pytest.mark.parametrize("connection_string", SUPPORTED_CONNECTION_STRINGS)
def test_ansi_codes_with_parameters(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)
    complex_ansi: str = (
        "\x1b[38;5;196mSELECT\x1b[0m \x1b[48;5;21m*\x1b[0m "
        "FROM \x1b[1;4;31musers\x1b[0m"
    )

    result: str = generator._extract_sql_code(complex_ansi)
    expected: str = "SELECT * FROM users"
    expected = get_expected_result(expected, connection_string=connection_string)

    compare_sql(result, expected)
    assert "\x1b" not in result


def test_dialect_extraction() -> None:
    postgres_gen = DatabaseQueryGenerator(SUPPORTED_CONNECTION_STRINGS[0])
    mssql_gen = DatabaseQueryGenerator(SUPPORTED_CONNECTION_STRINGS[1])
    mysql_gen = DatabaseQueryGenerator(SUPPORTED_CONNECTION_STRINGS[2])

    assert postgres_gen._dialect == "postgres"
    assert mssql_gen._dialect == "tsql"
    assert mysql_gen._dialect == "mysql"
