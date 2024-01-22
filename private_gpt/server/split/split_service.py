from typing import Literal

from injector import singleton
from llama_index import Document
from llama_index.node_parser import SentenceSplitter
from pydantic import BaseModel, Field


class Split(BaseModel):
    index: int
    object: Literal["split"]
    split: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of"
        ]
    )


@singleton
class SplitService:
    def texts_split(self, text: str, chunk_size: int) -> list[Split]:
        documents = [Document(text=text)]

        parser = SentenceSplitter(chunk_size=chunk_size)
        nodes = parser.get_nodes_from_documents(documents)

        return [
            Split(
                index=i,
                object="split",
                split=node.get_content(),
            )
            for i, node in enumerate(nodes)
        ]
