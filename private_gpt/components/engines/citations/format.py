import json
from collections.abc import Callable
from typing import Any

from llama_index.core.schema import NodeWithScore

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    ORIGINAL_END_TOKEN,
    ORIGINAL_START_TOKEN,
    convert_nodes_to_documents_list,
)
from private_gpt.components.ingest.metadata_helper import (
    MetadataChunk,
    MetadataKeys,
    MetadataNode,
)
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.settings.settings import settings


def format_llm_source_str(
    content: str,
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
) -> str:
    """Format node source for the LLM prompt."""
    if not generate_citations:
        return ""
    return f"{start_token}{content}{end_token}"


def format_llm_source(
    document: Document | None = None,
    node: NodeWithScore | None = None,
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
) -> str:
    assert document or node, "Either document or node must be provided"
    if document is None and node is not None:
        document = Document.from_node(node)
    return (
        format_llm_source_str(document.id, start_token, end_token, generate_citations)
        if document
        else ""
    )


def _format_documents_as_list(
    documents: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: Callable[[str], list[Any]] | None = None,
) -> tuple[list[Document], str]:
    """Format node IDs for the LLM prompt."""
    prefix = "Citation identifier " if generate_citations and documents else ""
    formatted_nodes = [
        f"{prefix}{format_llm_source(document=d, start_token=start_token, end_token=end_token, generate_citations=generate_citations)}\n---\nContent:\n"
        f"{d.text}\n===\n"
        for d in documents
    ]

    limited_nodes: list[Document] = []
    content_nodes: list[str] = []
    if token_limit is not None and tokenizer_fn is not None:
        total_tokens = 0
        for node, formatted_node in zip(documents, formatted_nodes, strict=False):
            node_tokens = len(tokenizer_fn(formatted_node))
            if total_tokens + node_tokens > token_limit:
                continue
            limited_nodes.append(node)
            content_nodes.append(formatted_node)
            total_tokens += node_tokens
    else:
        limited_nodes = documents
        content_nodes = formatted_nodes

    return limited_nodes, "".join(content_nodes)


def _format_documents_as_xml(
    documents: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: Callable[[str], list[Any]] | None = None,
) -> tuple[list[Document], str]:
    """Format doc IDs for the LLM prompt.

    Improvements:
    1. Groups information by document and adds filename as a header
    2. Sorts docs by document and absolute index for display
    3. Ensures most relevant content fits within token limits regardless of document
    4. Documents are ordered by the highest score of any doc within that document
    5. Nodes within each document are ordered by absolute index for coherent reading
    """
    # Sort docs by document and position for initial organization
    documents_by_position = sorted(
        documents,
        key=lambda d: (
            d.metadata.get(MetadataKeys.ARTIFACT_ID.value, ""),
            d.metadata.get(MetadataKeys.FILENAME.value, ""),
            d.metadata.get(MetadataChunk.ABS_IDX.value, 0),
        ),
    )

    # Group docs by document
    doc_groups: dict[str, list[Document]] = {}
    for n in documents_by_position:
        filename = n.metadata.get(MetadataKeys.FILENAME.value, "Unknown Document")
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(n)

    # Pre-format all docs and calculate token usage
    all_formatted_documents: list[tuple[str, Document, str, int]] = []

    for filename, documents in doc_groups.items():
        for d in documents:
            citation = format_llm_source(
                document=d,
                start_token=start_token,
                end_token=end_token,
                generate_citations=generate_citations,
            )
            doc_content = d.text or ""
            doc_content = doc_content.strip()

            if not doc_content:
                continue  # Skip empty content

            formatted_doc = (
                f"<node id='{citation}'>\n" f"{doc_content}\n" f"</node>\n\n"
                if generate_citations
                else f"<node>\n{doc_content}\n</node>\n\n"
            )
            doc_tokens = len(tokenizer_fn(formatted_doc)) if tokenizer_fn else 0
            all_formatted_documents.append((filename, d, formatted_doc, doc_tokens))

    # Handle token limit case
    if token_limit is not None and tokenizer_fn is not None:
        # Sort all docs by relevance score regardless of document
        all_formatted_documents_by_score = sorted(
            all_formatted_documents,
            key=lambda x: x[1].metadata.get(MetadataNode.SCORE.value, 0),
            reverse=True,
        )

        # Select docs based on score until we hit the token limit
        total_tokens = 0
        selected_docs = []
        added_doc_headers = set()

        for (
            filename,
            doc,
            formatted_doc,
            doc_tokens,
        ) in all_formatted_documents_by_score:
            doc_header = f"<document filename='{filename}' artifact_id='{doc.metadata.get(MetadataKeys.ARTIFACT_ID.value, 'unknown')}'>\n"
            doc_footer = "</document>\n\n"
            auxiliar_tokens = len(tokenizer_fn(doc_header)) + len(
                tokenizer_fn(doc_footer)
            )

            # Check if adding this doc (and possibly its header)
            # would exceed the token limit
            additional_tokens = doc_tokens
            if filename not in added_doc_headers:
                additional_tokens += auxiliar_tokens

            if total_tokens + additional_tokens > token_limit:
                continue

            # Add document header tokens if it's the first doc from this document
            if filename not in added_doc_headers:
                added_doc_headers.add(filename)
                total_tokens += auxiliar_tokens

            # Add the doc
            selected_docs.append((filename, doc, formatted_doc))
            total_tokens += doc_tokens

        # Group selected docs by document
        doc_to_docs: dict[str, list[tuple[Document, str]]] = {}
        for filename, doc, formatted_doc in selected_docs:
            if filename not in doc_to_docs:
                doc_to_docs[filename] = []
            doc_to_docs[filename].append((doc, formatted_doc))

        # Calculate max score for each document for ordering
        doc_max_scores = {}
        for filename, docs_list in doc_to_docs.items():
            doc_max_scores[filename] = max(
                doc.metadata.get(MetadataNode.SCORE.value, 0) for doc, _ in docs_list
            )

        # Order documents by their maximum doc score
        ordered_docs = sorted(
            doc_to_docs.keys(), key=lambda doc: doc_max_scores[doc], reverse=True
        )

        # Build the final content with proper document and doc ordering
        content_blocks = []
        all_limited_docs = []

        for filename in ordered_docs:
            docs_and_content = doc_to_docs[filename]

            # Sort docs within each document by absolute index
            docs_and_content.sort(
                key=lambda x: x[0].metadata.get(MetadataChunk.ABS_IDX.value, 0)
            )

            content_blocks.append(
                f"<document filename='{filename}' artifact_id='{docs_and_content[0][0].metadata.get(MetadataKeys.ARTIFACT_ID.value, 'unknown')}'>\n"
            )
            for doc, formatted_doc in docs_and_content:
                all_limited_docs.append(doc)
                content_blocks.append(formatted_doc)
            content_blocks.append("</document>\n\n")
    else:
        # If no token limit, include all docs sorted by document and position
        # Group by document and calculate max score per document
        doc_to_docs = {}
        doc_max_scores = {}

        for filename, document, formatted_doc, _ in all_formatted_documents:
            score = document.metadata.get(MetadataNode.SCORE.value, 0)
            if filename not in doc_to_docs:
                doc_to_docs[filename] = []
                doc_max_scores[filename] = score
            else:
                doc_max_scores[filename] = max(doc_max_scores[filename], score)

            doc_to_docs[filename].append((document, formatted_doc))

        # Order documents by their maximum doc score
        ordered_docs = sorted(
            doc_to_docs.keys(), key=lambda d: doc_max_scores[d], reverse=True
        )

        # Build the final content with proper document and doc ordering
        content_blocks = []
        all_limited_docs = []

        for filename in ordered_docs:
            docs_and_content = doc_to_docs[filename]
            docs_and_content.sort(
                key=lambda x: x[0].metadata.get(MetadataChunk.ABS_IDX.value, 0)
            )

            # Add document header
            content_blocks.append(
                f"<document filename='{filename}' artifact_id='{docs_and_content[0][0].metadata.get(MetadataKeys.ARTIFACT_ID.value, 'unknown')}'>\n"
            )
            for doc, formatted_doc in docs_and_content:
                all_limited_docs.append(doc)
                content_blocks.append(formatted_doc)
            content_blocks.append("</document>\n\n")

    formatted_content = "".join(content_blocks)
    all_limited_docs = sorted(
        all_limited_docs,
        key=lambda d: (
            d.metadata.get(MetadataKeys.FILENAME.value, ""),
            d.metadata.get(MetadataChunk.ABS_IDX.value, 0),
        ),
    )
    return all_limited_docs, formatted_content


def _format_documents_as_json(
    documents: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: Callable[[str], list[Any]] | None = None,
) -> tuple[list[Document], str]:
    documents_by_position = sorted(
        documents,
        key=lambda d: (
            d.metadata.get(MetadataKeys.ARTIFACT_ID.value, ""),
            d.metadata.get(MetadataKeys.FILENAME.value, ""),
            d.metadata.get(MetadataChunk.ABS_IDX.value, 0),
        ),
    )

    candidates: list[Document] = []
    for d in documents_by_position:
        if (d.text or "").strip():
            candidates.append(d)

    def _build(docs: list[Document]) -> tuple[list[Document], str]:
        doc_groups: dict[str, list[Document]] = {}
        for d in docs:
            filename = d.metadata.get(MetadataKeys.FILENAME.value, "Unknown Document")
            doc_groups.setdefault(filename, []).append(d)

        doc_max_scores: dict[str, float] = {
            filename: max(d.metadata.get(MetadataNode.SCORE.value, 0) for d in group)
            for filename, group in doc_groups.items()
        }

        ordered_filenames = sorted(
            doc_groups, key=lambda f: doc_max_scores[f], reverse=True
        )

        output: list[dict[str, Any]] = []
        result_docs: list[Document] = []

        for filename in ordered_filenames:
            group = sorted(
                doc_groups[filename],
                key=lambda d: d.metadata.get(MetadataChunk.ABS_IDX.value, 0),
            )
            nodes = []
            for d in group:
                citation = format_llm_source(
                    document=d,
                    start_token=start_token,
                    end_token=end_token,
                    generate_citations=generate_citations,
                )
                node_dict: dict[str, Any] = {}
                if generate_citations:
                    node_dict["id"] = citation
                node_dict["content"] = (d.text or "").strip()
                if not node_dict["content"]:
                    continue  # Skip empty content

                nodes.append(node_dict)
                result_docs.append(d)

            output.append(
                {
                    "filename": filename,
                    "artifact_id": group[0].metadata.get(
                        MetadataKeys.ARTIFACT_ID.value, "unknown"
                    ),
                    "nodes": nodes,
                }
            )

        result_docs = sorted(
            result_docs,
            key=lambda d: (
                d.metadata.get(MetadataKeys.FILENAME.value, ""),
                d.metadata.get(MetadataChunk.ABS_IDX.value, 0),
            ),
        )
        return result_docs, json.dumps(output)

    result_docs, content = _build(candidates)
    if token_limit is None or tokenizer_fn is None:
        return result_docs, content

    while candidates and len(tokenizer_fn(content)) > token_limit:
        # Remove the lowest-score candidate
        candidates.sort(key=lambda d: d.metadata.get(MetadataNode.SCORE.value, 0))
        candidates.pop(0)
        result_docs, content = _build(candidates)

    return result_docs, content


def _format_website_as_list(
    websites: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: Callable[[str], list[Any]] | None = None,
) -> tuple[list[Document], str]:
    """Format website IDs for the LLM prompt."""
    prefix = "Citation identifier " if generate_citations and websites else ""
    formatted_websites = [
        f"{prefix}{format_llm_source(document=w, start_token=start_token, end_token=end_token, generate_citations=generate_citations)}\n---\nContent:\n"
        f"{w.text}\n===\n"
        for w in websites
    ]

    limited_websites: list[Document] = []
    content_websites: list[str] = []
    if token_limit is not None and tokenizer_fn is not None:
        total_tokens = 0
        for website, formatted_website in zip(
            websites, formatted_websites, strict=False
        ):
            website_tokens = len(tokenizer_fn(formatted_website))
            if total_tokens + website_tokens > token_limit:
                continue
            limited_websites.append(website)
            content_websites.append(formatted_website)
            total_tokens += website_tokens
    else:
        limited_websites = websites
        content_websites = formatted_websites

    return limited_websites, "".join(content_websites)


def _format_document_by_type(
    document_type: str,
    documents: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: Callable[[str], list[Any]] | None = None,
) -> tuple[list[Document], str]:
    """Format documents by type for the LLM prompt."""
    selected_format_strategy = settings().chat.format_context_strategy
    match document_type:
        case "document":
            match selected_format_strategy:
                case "list":
                    return _format_documents_as_list(
                        documents,
                        start_token=start_token,
                        end_token=end_token,
                        generate_citations=generate_citations,
                        token_limit=token_limit,
                        tokenizer_fn=tokenizer_fn,
                    )
                case "xml":
                    return _format_documents_as_xml(
                        documents,
                        start_token=start_token,
                        end_token=end_token,
                        generate_citations=generate_citations,
                        token_limit=token_limit,
                        tokenizer_fn=tokenizer_fn,
                    )
                case "json":
                    return _format_documents_as_json(
                        documents,
                        start_token=start_token,
                        end_token=end_token,
                        generate_citations=generate_citations,
                        token_limit=token_limit,
                        tokenizer_fn=tokenizer_fn,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported format strategy: {selected_format_strategy}."
                        f" Supported strategies are: 'list', 'xml', 'json'."
                    )

        case "webpage":
            return _format_website_as_list(
                documents,
                start_token=start_token,
                end_token=end_token,
                generate_citations=generate_citations,
                token_limit=token_limit,
                tokenizer_fn=tokenizer_fn,
            )
        case _:
            raise ValueError(
                f"Unsupported document type: {type}. Supported types are: 'document'."
            )


def format_context(
    documents: list[Document] | None = None,
    nodes: list[NodeWithScore] | None = None,
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    generate_citations: bool = True,
    token_limit: int | None = None,
    tokenizer_fn: TokenizerFn | None = None,
) -> tuple[list[Document], str]:
    """Format context for the LLM prompt."""
    if documents is None and nodes is not None:
        tmp: list[NodeWithScore] = nodes or []
        docs = sorted(tmp, key=lambda x: float(x.score or 0), reverse=True)
        documents = convert_nodes_to_documents_list(docs)

    if not documents:
        return [], ""

    # Group documents by type
    documents_by_type: dict[str, list[Document]] = {}
    for doc in documents:
        doc_type: str = doc.type
        if doc_type not in documents_by_type:
            documents_by_type[doc_type] = []
        documents_by_type[doc_type].append(doc)

    # Format each group
    formatted_documents: list[Document] = []
    formatted_content = ""
    current_token_limit = token_limit or float("inf")

    for doc_type, current_docs in documents_by_type.items():
        limited_docs, content = _format_document_by_type(
            doc_type,
            current_docs,
            start_token=start_token,
            end_token=end_token,
            generate_citations=generate_citations,
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn,
        )
        potential_tokens_limit = len(tokenizer_fn(content)) if tokenizer_fn else None
        if (
            potential_tokens_limit is not None
            and potential_tokens_limit > current_token_limit
        ):
            continue

        formatted_documents.extend(limited_docs)
        formatted_content += content

    return formatted_documents, formatted_content
