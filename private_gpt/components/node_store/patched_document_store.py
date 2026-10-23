from typing import Any

from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.utils import json_to_doc

from private_gpt.components.readers.nodes.utils import (
    dict_to_tree_node,
)


def json_to_doc_tree(doc_dict: dict[str, Any]) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]

    if "-" not in doc_type:
        return json_to_doc(doc_dict)

    node_type, version = doc_type.split("-")
    return dict_to_tree_node(version, node_type, data_dict)


class PatchedKVDocumentStore(SimpleDocumentStore):
    """Document store that support tree nodes.

    In the original implementation, the document store
    only supports Llama Index documents. This class
    extends the document store to support tree nodes.

    """

    @property
    def docs(self) -> dict[str, BaseNode]:
        """Get all documents.

        Returns:
            Dict[str, BaseDocument]: documents

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_doc_tree(json) for key, json in json_dict.items()}

    def get_document(self, doc_id: str, raise_error: bool = True) -> BaseNode | None:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        json = self._kvstore.get(doc_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        return json_to_doc_tree(json)

    async def aget_document(
        self, doc_id: str, raise_error: bool = True
    ) -> BaseNode | None:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        json = await self._kvstore.aget(doc_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        return json_to_doc_tree(json)
