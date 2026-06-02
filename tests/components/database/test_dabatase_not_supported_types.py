from unittest.mock import patch

import pytest

from private_gpt.components.database.inspected_schema import (
    InspectedColumn,
    InspectedSchema,
    InspectedTable,
    TableType,
)
from private_gpt.components.tabular.database_query_generator import (
    DatabaseQueryGenerator,
)

# Connections strings only for create mock objects
DEFAULT_DATABASE_CONNECTIONS_STRING = ["mssql+pyodbc://user:pass@server:1433/testdb"]


def create_mock_column(name: str, type_: str) -> InspectedColumn:
    col = InspectedColumn()
    col.name = name
    col.type = type_
    return col


def create_mock_table(
    schema: str, table: str, columns: list[InspectedColumn]
) -> InspectedTable:
    tbl = InspectedTable()
    tbl.schema = schema
    tbl.name = table
    tbl.columns = columns
    tbl.table_type = TableType.TABLE
    return tbl


def create_mock_schema(name: str, tables: list[InspectedTable]) -> InspectedSchema:
    schema = InspectedSchema()
    schema.name = name
    for table in tables:
        schema.add_object(table)
    return schema


@pytest.mark.parametrize(
    "connection_string",
    DEFAULT_DATABASE_CONNECTIONS_STRING,
)
def test_select_star_excludes_geography(
    connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    columns = [
        create_mock_column("id", "INT"),
        create_mock_column("location", "NULL"),
        create_mock_column("name", "VARCHAR"),
    ]
    table = create_mock_table("dbo", "places", columns)
    schema = create_mock_schema("dbo", [table])

    original_sql = "SELECT * FROM dbo.places"

    with patch.object(generator, "_extract_database_schema", return_value=[schema]):
        result = generator._try_to_fix_unsupported_type_error(original_sql)

    expected_sql = (
        "SELECT\n" "  id\n" ", name\n" ", location.ToString()\n" "FROM\n" "  dbo.places"
    )

    assert result is not None
    assert result == expected_sql


@pytest.mark.parametrize(
    "connection_string",
    DEFAULT_DATABASE_CONNECTIONS_STRING,
)
def test_select_partial_columns_excludes_geography(connection_string: str) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    columns = [
        create_mock_column("id", "INT"),
        create_mock_column("location", "NULL"),
        create_mock_column("name", "VARCHAR"),
    ]
    table = create_mock_table("dbo", "places", columns)
    schema = create_mock_schema("dbo", [table])

    original_sql = "SELECT name, location FROM dbo.places"

    with patch.object(generator, "_extract_database_schema", return_value=[schema]):
        result = generator._try_to_fix_unsupported_type_error(original_sql)

    expected_sql = (
        "SELECT\n" "  name\n" ", location.ToString()\n" "FROM\n" "  dbo.places"
    )

    assert result is not None
    assert result == expected_sql


@pytest.mark.parametrize(
    "connection_string",
    DEFAULT_DATABASE_CONNECTIONS_STRING,
)
def test_select_columns_multiple_tables_excludes_geography(
    connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    # Define columns for places table
    place_columns = [
        create_mock_column("id", "INT"),
        create_mock_column("location", "NULL"),
        create_mock_column("name", "VARCHAR"),
    ]
    places_table = create_mock_table("dbo", "places", place_columns)

    # Define columns for employees table
    employee_columns = [
        create_mock_column("emp_id", "INT"),
        create_mock_column("emp_location", "NULL"),
        create_mock_column("emp_name", "VARCHAR"),
    ]
    employees_table = create_mock_table("dbo", "employees", employee_columns)

    # Schema including both tables
    schema = create_mock_schema("dbo", [places_table, employees_table])

    # Original query selecting specific columns from both tables
    original_sql = (
        "SELECT places.name, places.location, employees.emp_name, employees.emp_location "
        "FROM dbo.places, dbo.employees"
    )

    with patch.object(generator, "_extract_database_schema", return_value=[schema]):
        result = generator._try_to_fix_unsupported_type_error(original_sql)

    expected_sql = (
        "SELECT\n"
        "  places.name\n"
        ", places.location.ToString()\n"
        ", employees.emp_name\n"
        ", employees.emp_location.ToString()\n"
        "FROM\n"
        "  dbo.places\n"
        ", dbo.employees"
    )

    assert result is not None
    assert result == expected_sql


@pytest.mark.parametrize(
    "connection_string",
    DEFAULT_DATABASE_CONNECTIONS_STRING,
)
def test_select_columns_left_join_excludes_geography(
    connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    # Define columns for places table
    place_columns = [
        create_mock_column("id", "INT"),
        create_mock_column("location", "NULL"),
        create_mock_column("name", "VARCHAR"),
    ]
    places_table = create_mock_table("dbo", "places", place_columns)

    # Define columns for employees table
    employee_columns = [
        create_mock_column("emp_id", "INT"),
        create_mock_column("emp_location", "NULL"),
        create_mock_column("emp_name", "VARCHAR"),
    ]
    employees_table = create_mock_table("dbo", "employees", employee_columns)

    # Schema including both tables
    schema = create_mock_schema("dbo", [places_table, employees_table])

    # Original query with LEFT JOIN
    original_sql = (
        "SELECT places.name, places.location, employees.emp_name, employees.emp_location "
        "FROM dbo.places LEFT JOIN dbo.employees ON places.id = employees.emp_id"
    )

    with patch.object(generator, "_extract_database_schema", return_value=[schema]):
        result = generator._try_to_fix_unsupported_type_error(original_sql)

    expected_sql = (
        "SELECT\n"
        "  places.name\n"
        ", places.location.ToString()\n"
        ", employees.emp_name\n"
        ", employees.emp_location.ToString()\n"
        "FROM\n"
        "  (dbo.places\n"
        "LEFT JOIN dbo.employees ON (places.id = employees.emp_id))"
    )

    assert result is not None
    assert result == expected_sql


@pytest.mark.parametrize(
    "connection_string",
    DEFAULT_DATABASE_CONNECTIONS_STRING,
)
def test_select_columns_right_join_excludes_geography(
    connection_string: str,
) -> None:
    generator = DatabaseQueryGenerator(connection_string)

    # Define columns for places table
    place_columns = [
        create_mock_column("id", "INT"),
        create_mock_column("location", "NULL"),
        create_mock_column("name", "VARCHAR"),
    ]
    places_table = create_mock_table("dbo", "places", place_columns)

    # Define columns for employees table
    employee_columns = [
        create_mock_column("emp_id", "INT"),
        create_mock_column("emp_location", "NULL"),
        create_mock_column("emp_name", "VARCHAR"),
    ]
    employees_table = create_mock_table("dbo", "employees", employee_columns)

    # Schema including both tables
    schema = create_mock_schema("dbo", [places_table, employees_table])

    # Original query with RIGHT JOIN
    original_sql = (
        "SELECT places.name, places.location, employees.emp_name, employees.emp_location "
        "FROM dbo.places RIGHT JOIN dbo.employees ON places.id = employees.emp_id"
    )

    with patch.object(generator, "_extract_database_schema", return_value=[schema]):
        result = generator._try_to_fix_unsupported_type_error(original_sql)

    expected_sql = (
        "SELECT\n"
        "  places.name\n"
        ", places.location.ToString()\n"
        ", employees.emp_name\n"
        ", employees.emp_location.ToString()\n"
        "FROM\n"
        "  (dbo.places\n"
        "RIGHT JOIN dbo.employees ON (places.id = employees.emp_id))"
    )

    assert result is not None
    assert result == expected_sql
