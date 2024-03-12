from collections.abc import Generator
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.chroma import ChromaVectorStore  # type: ignore


def chunk_list(
    lst: list[BaseNode], max_chunk_size: int
) -> Generator[list[BaseNode], None, None]:
    """Yield successive max_chunk_size-sized chunks from lst.

    Args:
        lst (List[BaseNode]): list of nodes with embeddings
        max_chunk_size (int): max chunk size

    Yields:
        Generator[List[BaseNode], None, None]: list of nodes with embeddings
    """
    for i in range(0, len(lst), max_chunk_size):
        yield lst[i : i + max_chunk_size]


class BatchedChromaVectorStore(ChromaVectorStore):  # type: ignore
    """Chroma vector store, batching additions to avoid reaching the max batch limit.

    In this vector store, embeddings are stored within a ChromaDB collection.

    During query time, the index uses ChromaDB to query for the top
    k most similar nodes.

    Args:
        chroma_client (from chromadb.api.API):
            API instance
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    """

    chroma_client: Any | None

    def __init__(
        self,
        chroma_client: Any,
        chroma_collection: Any,
        host: str | None = None,
        port: str | None = None,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        collection_kwargs: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(
            chroma_collection=chroma_collection,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            collection_kwargs=collection_kwargs or {},
        )
        self.chroma_client = chroma_client

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes to index, batching the insertion to avoid issues.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
            add_kwargs: _
        """
        if not self.chroma_client:
            raise ValueError("Client not initialized")

        if not self._collection:
            raise ValueError("Collection not initialized")

        max_chunk_size = self.chroma_client.max_batch_size
        node_chunks = chunk_list(nodes, max_chunk_size)

        all_ids = []
        for node_chunk in node_chunks:
            embeddings = []
            metadatas = []
            ids = []
            documents = []
            for node in node_chunk:
                embeddings.append(node.get_embedding())
                metadatas.append(
                    node_to_metadata_dict(
                        node, remove_text=True, flat_metadata=self.flat_metadata
                    )
                )
                ids.append(node.node_id)
                documents.append(node.get_content(metadata_mode=MetadataMode.NONE))

            self._collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            all_ids.extend(ids)

        return all_ids
