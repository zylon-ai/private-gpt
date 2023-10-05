from llama_index.storage.docstore import BaseDocumentStore

from private_gpt.open_ai.extensions.context_files import ContextFiles


def get_context_nodes(
    context_files: ContextFiles, docstore: BaseDocumentStore
) -> list[str]:
    # TODO filter by the info provided in context_files provided
    return [node.node_id for node in docstore.docs.values()]
