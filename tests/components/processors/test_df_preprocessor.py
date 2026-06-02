from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from private_gpt.components.ingest.processors.df_preprocessor import (
    DataFramePreprocessor,
)


@pytest.fixture
def processor() -> DataFramePreprocessor:
    return DataFramePreprocessor(try_cast_to_numeric=True, try_cast_to_datetime=True)


def test_convert_column(processor: DataFramePreprocessor) -> None:
    # Test numeric conversion
    numeric_series = pd.Series(["1", "2", "3"])
    expected = pd.Series([1, 2, 3])
    assert_series_equal(processor._convert_column(numeric_series), expected)

    # Test datetime conversion
    date_series = pd.Series(["2024-01-01", "2024-01-02"])
    expected = pd.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)])
    assert_series_equal(processor._convert_column(date_series), expected)

    # Test string stripping
    string_series = pd.Series([" test ", "hello  ", "  world"])
    expected = pd.Series(["test", "hello", "world"])
    assert_series_equal(processor._convert_column(string_series), expected)

    # Test numeric conversion
    numeric_series = pd.Series(["1", "2.2", "3000"])
    expected = pd.Series([1, 2.2, 3000])
    assert_series_equal(processor._convert_column(numeric_series), expected)


def test_header_detection(processor: DataFramePreprocessor) -> None:
    df = pd.DataFrame([[1, 2], [3, 4]])
    assert processor._is_default_header(df)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert not processor._is_default_header(df)
    df = pd.DataFrame([[" Name ", "Age"], ["John", 25], ["Jane", 30]])
    assert processor._is_inferred_header(df)


def test_header_renaming(processor: DataFramePreprocessor) -> None:
    df = pd.DataFrame(
        [
            ["Name", "Age", "Name", "", "nan"],
            ["John", 25, "Student", "A", "X"],
            ["Jane", 30, "Teacher", "B", "Y"],
        ]
    )
    result = processor.preprocess_table(df)
    assert "Name_2" in result.columns
    assert "Unknown_1" in result.columns
    assert "Unknown_2" in result.columns


def test_complete_preprocessing(processor: DataFramePreprocessor) -> None:
    df = pd.DataFrame(
        [
            ["Name ", "Age", "Date", "Name", ""],
            [" John ", "25", "2024-01-01", "Student", "A"],
            [" Jane ", "30", "2024-01-02", "Teacher", "B"],
        ]
    )

    result = processor.preprocess_table(df)
    assert pd.api.types.is_numeric_dtype(result["Age"])
    assert pd.api.types.is_datetime64_dtype(result["Date"])
    assert all(result["Name_1"].str.strip() == result["Name_1"])
    assert "Name_2" in result.columns
    assert "Unknown" in result.columns


def test_edge_cases(processor: DataFramePreprocessor) -> None:
    # Empty DataFrame
    df_empty = pd.DataFrame()
    result = processor.preprocess_table(df_empty)
    assert result.empty

    # Single row DataFrame
    df_single = pd.DataFrame([["a", "b"]])
    result = processor.preprocess_table(df_single)
    assert len(result) == 1

    # Single row with missing values
    df_single = pd.DataFrame({"A": [None], "B": [1]})
    result = processor.preprocess_table(df_single)
    assert len(result.columns) == 1

    # All missing values except one
    df = pd.DataFrame({"col": [None] * 9 + [1]})
    result = processor.preprocess_table(df)
    assert result["col"].isna().sum() == 0
    assert result["col"].dtype == np.float64


def test_empty_first_cell_header_detection(processor: DataFramePreprocessor) -> None:
    # Test with space in first cell
    df1 = pd.DataFrame([[" ", "Age", "City"], ["John", 25, "NY"], ["Jane", 30, "LA"]])
    result1 = processor.preprocess_table(df1)
    assert list(result1.columns) == ["Unknown", "Age", "City"]
    assert len(result1) == 2  # Should remove header row

    # Test with empty string in first cell
    df2 = pd.DataFrame([["", "Column2", "Column3"], ["Data1", "Data2", "Data3"]])
    result2 = processor.preprocess_table(df2)
    assert list(result2.columns) == ["Unknown", "Column2", "Column3"]
    assert len(result2) == 1


def test_empty_rows_and_columns(processor: DataFramePreprocessor) -> None:
    # Test empty column
    df1 = pd.DataFrame({"A": ["", "", ""], "B": [1, 2, 3], "C": ["x", "y", "z"]})
    result1 = processor.preprocess_table(df1)
    assert "A" not in result1.columns
    assert len(result1.columns) == 2

    # Test empty row
    df2 = pd.DataFrame([["a", "b", "c"], ["", "", ""], ["d", "e", "f"]])
    result2 = processor.preprocess_table(df2)
    assert len(result2) == 2

    # Test both empty rows and columns
    df3 = pd.DataFrame([["", "b", ""], ["", "", ""], ["", "e", ""]])
    result3 = processor.preprocess_table(df3)
    assert len(result3.columns) == 1
    assert len(result3) == 1

    df4 = pd.DataFrame([["", "", ""], ["b", "", "e"], ["", "", ""]])
    result4 = processor.preprocess_table(df4)
    assert len(result4.columns) == 2
    assert len(result4) == 1


def test_mixed_empty_values(processor: DataFramePreprocessor) -> None:
    df = pd.DataFrame(
        [
            ["nan", "Age", "null"],
            ["John", 25, "NY"],
            ["Jane", 30, "LA"],
            ["n/a", "n/a", "n/a"],
        ]
    )
    result = processor.preprocess_table(df)
    assert "Unknown_1" in result.columns
    assert len(result) == 2


def test_numeric_conversion() -> None:
    """Test improved numeric conversion with mixed data types."""
    processor = DataFramePreprocessor()

    # Test with mixed numeric and non-numeric (should stay as string)
    mixed_series = pd.Series(["1", "2", "hello", "4"])
    result = processor._convert_column(mixed_series)
    assert result.dtype == object  # Should remain as string due to low numeric ratio

    # Test with mostly numeric (should convert)
    mostly_numeric = pd.Series(["1", "2", "3", "4", "hello"])
    result = processor._convert_column(mostly_numeric)
    assert pd.api.types.is_numeric_dtype(result)


def test_datetime_conversion() -> None:
    """Test improved datetime conversion with validation."""
    processor = DataFramePreprocessor()

    # Test with non-datetime strings (should not convert)
    non_dates = pd.Series(["hello", "world", "test"])
    result = processor._convert_column(non_dates)
    assert not pd.api.types.is_datetime64_any_dtype(result)

    # Test with mixed dates and non-dates (should not convert due to ratio)
    mixed_dates = pd.Series(["2024-01-01", "hello", "world"])
    result = processor._convert_column(mixed_dates)
    assert not pd.api.types.is_datetime64_any_dtype(result)


def test_empty_value_handling() -> None:
    """Test handling of various empty value formats."""
    processor = DataFramePreprocessor()

    df = pd.DataFrame(
        [
            ["Name", "Age", "City"],
            ["John", "25", "NYC"],
            ["", "nan", "null"],
            ["Jane", "30", "LA"],
        ]
    )

    result = processor.preprocess_table(df)

    # Should have proper headers
    assert list(result.columns) == ["Name", "Age", "City"]

    # Should remove the empty row
    assert len(result) == 2

    # Age should be numeric
    assert pd.api.types.is_numeric_dtype(result["Age"])


def test_nullable_integer_conversion() -> None:
    """Test conversion to nullable integer types."""
    processor = DataFramePreprocessor()

    # Test integer conversion with missing values
    int_with_na = pd.Series([1, 2, None, 4])
    result = processor._convert_column(int_with_na)
    assert pd.api.types.is_numeric_dtype(result)


def test_robust_type_inference() -> None:
    """Test robust type inference with edge cases."""
    processor = DataFramePreprocessor(min_numeric_ratio=0.61, min_datetime_ratio=0.61)

    # Test with borderline numeric data
    borderline_numeric = pd.Series(["1", "2", "3", "invalid", "invalid"])
    result = processor._convert_column(borderline_numeric)
    assert result.dtype == object  # Should not convert due to ratio

    # Test with sufficient numeric data
    sufficient_numeric = pd.Series(["1", "2", "3", "4", "invalid"])
    result = processor._convert_column(sufficient_numeric)
    assert pd.api.types.is_numeric_dtype(result)


def test_complex_preprocessing_scenario():
    """Test complex preprocessing scenario with multiple issues."""
    processor = DataFramePreprocessor()

    df = pd.DataFrame(
        [
            [
                "",
                "Sales ",
                "Date",
                "Sales",
                "Notes",
            ],  # Empty first cell, duplicates, whitespace
            ["Q1", " 1000 ", "2024-01-01", "1500", "Good"],
            ["", "", "", "", ""],  # Empty row
            ["Q2", "2000", "2024-02-01", "2500", "Better"],
            ["Q3", "nan", "invalid_date", "3000", ""],  # Mixed data types
        ]
    )

    result = processor.preprocess_table(df)

    # Check headers are properly handled
    expected_columns = ["Unknown", "Sales_1", "Date", "Sales_2", "Notes"]
    assert list(result.columns) == expected_columns

    # Check empty row is removed
    assert len(result) == 3

    # Check numeric conversion
    assert not pd.api.types.is_numeric_dtype(
        result["Sales_1"]
    )  # numeric ratio is lower than 0.8
    assert pd.api.types.is_numeric_dtype(result["Sales_2"])
