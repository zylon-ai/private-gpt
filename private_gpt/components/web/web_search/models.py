from typing import Any

from pydantic import BaseModel


class WebSearchResult(BaseModel):
    idx: int | None = None
    title: str
    url: str
    favicon_url: str | None = None
    description: str
    age: str | None = None
    content: str | None = None
    content_type: str | None = None
    tokens: int = 0
    metadata: dict[str, Any] | None = None
    is_relevant: bool = False
    is_in_error: bool = False

    def __str__(self) -> str:
        result = ""

        if self.idx is not None:
            result += f"{self.idx}. "

        result += f"{self.title}\n"
        result += f"Description: {self.description}\n"
        result += f"URL: {self.url}\n"

        if self.age:
            result += f"Age: {self.age}\n"

        if self.metadata:
            result += f"Metadata: {self.metadata}\n"

        if self.content:
            result += f"Content: {self.content}\n"

        return result
