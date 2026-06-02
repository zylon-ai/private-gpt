import builtins
import enum
import re
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from private_gpt.components.ingest.processors.df_preprocessor import (
    VALID_DATETIME_FORMATS,
)
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode
from private_gpt.utils.dataframe import df_to_minimal_markdown


class NpEncoder:
    _complex_numeric_dtypes = (np.datetime64, np.complexfloating)
    _missing_values = (pd.NA, np.NAN, np.NaN, pd.NaT)

    def encode(self, obj: Any) -> Any:
        if isinstance(obj, self._complex_numeric_dtypes):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if any(np.issubdtype(obj.dtype, i) for i in self._complex_numeric_dtypes):
                return obj.astype(str).tolist()
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, list | tuple):
            return [self.encode(item) for item in obj]
        elif pd.isna(obj):
            return None
        return obj

    def decode(self, obj: Any) -> Any:
        if isinstance(obj, str):
            # First try to convert to float/int
            try:
                # Check if it's a number with decimal point
                if "." in obj:
                    return float(obj)
                return int(obj)
            except ValueError:
                # Only try timestamp conversion if the string matches date-like patterns
                if any(re.match(pattern, obj) for pattern in VALID_DATETIME_FORMATS):
                    try:
                        return pd.Timestamp(obj)
                    except ValueError:
                        pass
                return obj
        elif isinstance(obj, int | float):
            return obj
        elif isinstance(obj, list):
            return [self.decode(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.decode(value) for key, value in obj.items()}
        return obj


class TableRepresentation(enum.StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"
    KEY_VALUE = "key_value"


DEFAULT_TABLE_ROW_REPRESENTATION = {
    # Representation that see LLM
    TreeMetadataMode.RAG: TableRepresentation.MARKDOWN,
    TreeMetadataMode.LLM: TableRepresentation.MARKDOWN,
    # Representation that we store in the embedding vector
    TreeMetadataMode.EMBED: TableRepresentation.KEY_VALUE,
}

DEFAULT_TABLE_REPRESENTATION = {
    # Representation that see LLM
    TreeMetadataMode.RAG: TableRepresentation.MARKDOWN,
    TreeMetadataMode.LLM: TableRepresentation.MARKDOWN,
    # Representation that we store in the embedding vector
    TreeMetadataMode.EMBED: TableRepresentation.MARKDOWN,
}


class TableRowNode(TreeNode):
    header: list[str] = Field(description="Header of the table row.")
    content: list[Any] = Field(description="Content of the table row.")

    class Meta(BaseModel, arbitrary_types_allowed=True):
        header: list[str]
        content: list[Any]

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.ALL
    ) -> str:
        df = (
            pd.DataFrame(
                [self.content] if self.content else [], columns=self.header, dtype=str
            )
            if self.header
            else pd.DataFrame()
        )

        table_representation = DEFAULT_TABLE_ROW_REPRESENTATION.get(
            metadata_mode, TableRepresentation.MARKDOWN
        )
        content = ""
        itself = metadata_mode == TreeMetadataMode.NONE
        match table_representation:
            case TableRepresentation.MARKDOWN:
                is_first_row = self.is_first_row()
                is_last_row = not is_first_row and self.is_last_row()

                # Remove header when it is not the first row
                # and representation is not only for itself
                remove_header = itself or not is_first_row
                # Add footer separator when it is the last row
                # and representation is not only for itself
                add_footer = not itself and is_last_row

                # Convert the DataFrame to a minimal Markdown table
                markdown = df_to_minimal_markdown(
                    df.fillna(""), allow_empty=False
                ).strip()

                if remove_header:
                    # Remove the header and separator
                    markdown = markdown.split("\n", 2)[-1]
                else:
                    if not itself and self.is_first_row():
                        # Add a header separator
                        markdown = "\n\n" + markdown

                if add_footer:
                    # Add a footer separator
                    markdown += "\n\n"

                content = markdown.replace("\n\n", "\n")
            case TableRepresentation.JSON:
                content = df.to_json(orient="records", lines=True)
            case TableRepresentation.KEY_VALUE:
                content = ", ".join(
                    [
                        f"{header}: {value}"
                        for header, value in zip(
                            self.header, self.content, strict=False
                        )
                    ]
                )

        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        row_content = content + "\n"
        return metadata_str + row_content

    def model_dump(
        self,
        *,
        include_parent: bool = False,
        include_children: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        encoder = NpEncoder()
        d = super().model_dump(
            include_parent=include_parent, include_children=include_children, **kwargs
        )
        d["content"] = encoder.encode(d["content"])
        return d

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        encoder = NpEncoder()
        data["content"] = encoder.decode(data["content"])
        return super().from_dict(data, **kwargs)

    def set_content(self, value: Any) -> None:
        if not isinstance(value, TableRowNode.Meta):
            raise ValueError(f"Expected TableRowNode.Meta, got {type(value)}")
        if len(value.header) != len(value.content):
            raise ValueError(
                f"Header and content length mismatch: {len(value.header)} != {len(value.content)}"
            )

        # Store content
        self.header = value.header
        self.content = value.content

    def is_first_row(self) -> bool:
        if not self.parent:
            return False

        siblings = self.parent.children
        if not siblings:
            return False

        # Validate idx - defensive check for partial loading
        current_index = self.idx if 0 <= self.idx < len(siblings) else None
        if current_index is not None and siblings[current_index] is not self:
            current_index = None

        # Fast path: if validated idx is 0, we're first
        if current_index == 0:
            return True

        # If we don't have current index, search with early termination
        if current_index is None:
            for i, sibling in enumerate(siblings):
                if isinstance(sibling, TableRowNode):
                    if sibling is self:
                        current_index = i
                        break
                    else:
                        # Found another TableRowNode before us
                        return False

        if current_index == 0:
            return True

        # Use validated current_index for left-sibling check
        if current_index is not None and current_index > 0:
            left_sibling = siblings[current_index - 1]
            return not isinstance(left_sibling, TableRowNode)

        # Fallback: if we couldn't determine current_index, assume we're not first
        return False

    def is_last_row(self) -> bool:
        if not self.parent:
            return False

        siblings = self.parent.children
        if not siblings:
            return False

        # Validate idx - defensive check for partial loading
        current_index = self.idx if 0 <= self.idx < len(siblings) else None
        if current_index is not None and siblings[current_index] is not self:
            current_index = None

        # Fast path: if validated idx is last position, we're last
        if current_index == len(siblings) - 1:
            return True

        # If we don't have current index, search backwards with early termination
        if current_index is None:
            for i in range(len(siblings) - 1, -1, -1):
                sibling = siblings[i]
                if isinstance(sibling, TableRowNode):
                    if sibling is self:
                        current_index = i
                        break
                    else:
                        # Found another TableRowNode after us (searching backwards)
                        return False

        if current_index == len(siblings) - 1:
            return True

        # Use validated current_index for right-sibling check
        if current_index is not None and current_index < len(siblings) - 1:
            right_sibling = siblings[current_index + 1]
            return not isinstance(right_sibling, TableRowNode)

        # Fallback: if we couldn't determine current_index, assume we're not first
        return False


class TableNode(TreeNode, arbitrary_types_allowed=True):
    df: pd.DataFrame = Field(
        description="Pandas DataFrame containing the table data.",
    )
    description: str | None = Field(
        default=None,
        description="Description of the table.",
    )

    class Meta(BaseModel, arbitrary_types_allowed=True):
        dataframe: pd.DataFrame
        summary: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        encoder = NpEncoder()
        model = super().model_dump(exclude={"df"}, **kwargs)
        model["df"] = self.df.to_dict(orient="split")
        if "data" in model["df"]:
            model["df"]["data"] = encoder.encode(model["df"]["data"])
        return model

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        encoder = NpEncoder()
        for name, value in data.items():
            if isinstance(value, dict) and name == "df" and "data" in value:
                value["data"] = encoder.decode(value["data"])
                data[name] = pd.DataFrame(**value)
        obj = super().from_dict(data, **kwargs)
        if not isinstance(obj, TableNode):
            raise ValueError(f"Expected TableNode, got {type(obj)}")
        return obj

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.ALL
    ) -> str:
        if metadata_mode == TreeMetadataMode.RAG:
            return ""

        table_representation = DEFAULT_TABLE_REPRESENTATION.get(
            metadata_mode, TableRepresentation.MARKDOWN
        )
        content = ""
        match table_representation:
            case TableRepresentation.MARKDOWN:
                df = self.df.astype(str)
                df = df.where(pd.notnull(df), "")
                df = df.replace(str(pd.NA), "")
                content = df_to_minimal_markdown(df) + "\n"
            case TableRepresentation.JSON:
                content = self.df.to_json(orient="records", lines=True)
            case TableRepresentation.KEY_VALUE:
                content = "".join(
                    child.get_content_internal(metadata_mode=metadata_mode)
                    for child in self.children
                )

        metadata_str = ""
        description = ""
        if metadata_mode != TreeMetadataMode.NONE:
            metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
            description = (
                f"Table description: \n{self.description}\n" if self.description else ""
            )

        content = f"Content: \n{content}" if self.description else content
        return metadata_str + description + content

    def set_content(self, value: Any) -> None:
        if not isinstance(value, TableNode.Meta):
            raise ValueError(f"Expected TableNode.Meta, got {type(value)}")
        if len(value.dataframe) == 0:
            raise ValueError("Empty dataframe provided.")

        # Store content
        self.df = value.dataframe
        self.description = value.summary

    def is_row_compatible(self, row: TableRowNode) -> bool:
        return all(
            col1 == col2
            for col1, col2 in zip(self.df.columns, row.header, strict=False)
        )

    def add_row(self, row: list[Any]) -> None:
        if len(row) != len(self.df.columns):
            raise ValueError(
                f"Row length mismatch: {len(row)} != {len(self.df.columns)}"
            )
        self.df.loc[len(self.df)] = row
