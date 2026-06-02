from __future__ import annotations

import csv
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pandas import Series


VALID_DATETIME_FORMATS = [
    r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
    r"^\d{2}/\d{2}/\d{4}$",  # MM/DD/YYYY
    r"^\d{2}-\d{2}-\d{4}$",  # MM-DD-YYYY
    r"^\d{4}/\d{2}/\d{2}$",  # YYYY/MM/DD
    r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # M/D/YY or MM/DD/YYYY
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # YYYY-MM-DD HH:MM:SS
]


class DataFramePreprocessor:
    """Handles DataFrame preprocessing with type inference and header detection."""

    def __init__(
        self,
        missing_threshold: float = 0.9,
        try_cast_to_numeric: bool = True,
        try_cast_to_datetime: bool = True,
        min_numeric_ratio: float = 0.8,
        min_datetime_ratio: float = 0.8,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.min_numeric_ratio = min_numeric_ratio
        self.min_datetime_ratio = min_datetime_ratio
        self._try_cast_to_numeric = try_cast_to_numeric
        self._try_cast_to_datetime = try_cast_to_datetime
        self._empty_values = frozenset(
            {"nan", "none", "null", "", "undefined", "n/a", "na", "nat"}
        )
        self._datetime_patterns = VALID_DATETIME_FORMATS

    def _is_numeric_column(self, column: Series[Any]) -> bool:
        """Check if column should be treated as numeric."""
        if not self._try_cast_to_numeric:
            return False

        # Remove empty values for testing
        non_empty = column.dropna()
        if len(non_empty) == 0:
            return False

        # Check if enough values can be converted to numeric
        numeric_count = 0
        for value in non_empty:
            with suppress(Exception):
                pd.to_numeric(str(value))
                numeric_count += 1

        return numeric_count / len(non_empty) >= self.min_numeric_ratio

    def _is_datetime_column(self, column: Series[Any]) -> bool:
        """Check if column should be treated as datetime."""
        if not self._try_cast_to_datetime:
            return False

        # Remove empty values for testing
        non_empty = column.dropna()
        if len(non_empty) == 0:
            return False

        # First check if values match common datetime patterns
        pattern_matches = 0
        for value in non_empty:
            str_value = str(value).strip()
            if any(re.match(pattern, str_value) for pattern in self._datetime_patterns):
                pattern_matches += 1

        # If not enough pattern matches, it's likely not a datetime column
        if pattern_matches / len(non_empty) < self.min_datetime_ratio:
            return False

        # Try to parse datetime
        datetime_count = 0
        for value in non_empty:
            with suppress(Exception):
                parsed = pd.to_datetime(str(value), errors="raise")
                # Additional validation: ensure it's a reasonable date
                if pd.Timestamp.min < parsed < pd.Timestamp.max:
                    datetime_count += 1

        return datetime_count / len(non_empty) >= self.min_datetime_ratio

    def _convert_column(self, column: Series[Any]) -> Series[Any]:
        """Convert column to most appropriate data type."""
        missing_ratio = (
            float(column.isna().sum()) / len(column) if len(column) > 0 else 0.0
        )
        if missing_ratio > self.missing_threshold:
            return column

        # Try numeric conversion
        if self._try_cast_to_numeric:
            with suppress(Exception):
                numeric_column = pd.to_numeric(column, errors="coerce")
                # Check if conversion was successful for most values
                if not numeric_column.isna().all():
                    non_null_original = column.dropna()
                    non_null_converted = numeric_column.dropna()
                    if (
                        len(non_null_converted) / len(non_null_original)
                        >= self.min_numeric_ratio
                    ):
                        return numeric_column

        # Try datetime conversion
        if self._try_cast_to_datetime and self._is_datetime_column(column):
            with suppress(Exception):
                datetime_column = pd.to_datetime(column, errors="coerce")
                if not datetime_column.isna().all():
                    return datetime_column

        # Return as cleaned string
        return column.astype(str).str.strip()

    def _is_default_header(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has default numeric headers."""
        return all(str(df.columns[i]) == str(i) for i in range(len(df.columns)))

    def _is_inferred_header(self, df: pd.DataFrame) -> bool:
        """Determine if first row is likely a header."""
        if df.empty or df.shape[0] == 1:
            return False

        first_row = df.iloc[0]

        # Type check
        if not all(isinstance(x, str | int | float) for x in first_row):
            return False

        # If first cell is an empty string, it's likely a header
        if str(first_row[0]).strip().lower() in self._empty_values:
            return True

        # Compare types without first row
        with suppress(Exception):
            with_first = df.iloc[:10].apply(self._convert_column).dtypes
            without_first = df.iloc[1:10].apply(self._convert_column).dtypes
            if not with_first.equals(without_first):
                return True

        # CSV header detection as fallback
        with suppress(Exception):
            sample = df.iloc[: min(10, len(df))].to_csv(index=False, header=False)
            return csv.Sniffer().has_header(sample)

        return False

    def _convert_into_str(self, columns: list[str] | None) -> list[str]:
        """Convert headers into strings if they are not already."""
        if columns is None:
            return []

        return [str(col) for col in columns]

    def _rename_empty_headers(self, columns: list[str]) -> list[str]:
        """Rename empty or invalid column names."""
        processed_columns = []
        for col in columns:
            original = str(col)
            col_str = original.strip()

            if (not col_str) or (col_str.lower() in self._empty_values):
                new_name = "Unknown"
                processed_columns.append(new_name)
            else:
                processed_columns.append(col_str)

        return processed_columns

    def _rename_duplicate_headers(self, columns: list[str]) -> list[str]:
        """Rename duplicate column names."""
        seen: dict[str, int] = {}
        processed_columns = []

        for i, col in enumerate(columns):
            if col not in seen:
                seen[col] = 1
                processed_columns.append(col)
                continue

            if seen[col] == 1:
                for j in range(i - 1, -1, -1):
                    if columns[j] == col:
                        processed_columns[j] = f"{col}_{seen[col]}"
                        break
            seen[col] += 1
            processed_columns.append(f"{col}_{seen[col]}")

        return processed_columns

    def _remove_empty_rows_and_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows and columns that are entirely empty values."""
        if df.empty:
            return df

        # Convert to string and check for empty values
        empty_mask = (
            df.astype(str)
            .apply(lambda x: x.str.strip().str.lower())
            .isin(self._empty_values)
        )
        df_cleaned = df.mask(empty_mask)

        # Remove rows where all values are empty/NaN
        non_empty_rows = ~df_cleaned.isna().all(axis=1)
        df = df.loc[non_empty_rows]

        # Remove columns where all values are empty/NaN
        non_empty_cols = ~df_cleaned.isna().all()
        df = df.loc[:, non_empty_cols]

        return df.reset_index(drop=True)

    def preprocess_table_data(
        self, rows: list[list[str]], headers: list[str] | None
    ) -> pd.DataFrame:
        """Preprocess DataFrame with type conversion and header detection."""
        headers = self._rename_empty_headers(headers or [])
        headers = self._rename_duplicate_headers(headers)
        headers = self._convert_into_str(headers)

        df = pd.DataFrame(rows, columns=headers or None)
        return self.preprocess_table(df)

    def preprocess_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame with type conversion and header detection."""
        if df.empty:
            return df

        # Handle headers if needed
        if not self._is_default_header(df):
            pass
        elif self._is_inferred_header(df):
            df_copy = df.copy()
            columns = list(df_copy.iloc[0])
            columns = self._rename_empty_headers(columns)
            columns = self._rename_duplicate_headers(columns)
            columns = self._convert_into_str(columns)
            df_copy.columns = pd.Index(columns)
            df_copy = df_copy[1:].reset_index(drop=True)
            df = df_copy

        # Remove empty rows and columns
        df = self._remove_empty_rows_and_columns(df)

        # Convert data types
        for col in df.columns:
            df[col] = self._convert_column(df[col])

        # Convert columns to string
        df.columns = pd.Index(self._convert_into_str(list(df.columns)))

        return df
