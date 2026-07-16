import json
import logging
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.schema import MetadataMode, NodeWithScore

from private_gpt.components.chat.processors.chat_history.memory.utils.splitting import (
    get_user_blocks,
)
from private_gpt.components.engines.citations.types import Citation, Document
from private_gpt.components.ingest.metadata_helper import (
    MetadataFlags,
    MetadataHelper,
    MetadataNode,
)
from private_gpt.di import get_global_injector
from private_gpt.events.models import SourceBlock, ThinkingBlock
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.components.chunk.models import SourceType
    from private_gpt.components.engines.citations.term_extractor import (
        TextAnalyzer as TextAnalyzerType,
    )

try:
    from private_gpt.components.engines.citations.term_extractor import (
        TextAnalyzer as ImportedTextAnalyzer,
    )
except ImportError:
    TextAnalyzer: type["TextAnalyzerType"] | None = None
else:
    TextAnalyzer = ImportedTextAnalyzer

logger = logging.getLogger(__name__)

ORIGINAL_START_TOKEN = "["
ORIGINAL_END_TOKEN = "]"

NUMERICAL_SHORTER_ID = settings().chat.numerical_shorter_citations
SHORTER_ID_LENGTH = 4
SHORTER_ID_FIELD = MetadataFlags.SHORTER_ID.value

DEFAULT_UNK_TOKEN = "UNK"
DEFAULT_SPLIT_CITATION_TOKEN = ","


def analyze_texts(
    texts: list[str],
    min_length: int | None = SHORTER_ID_LENGTH,
    max_length: int | None = SHORTER_ID_LENGTH,
    max_terms: int = 1,
    langs: set[str] | None = None,
) -> dict[int, list[str]]:
    """Convenience function to analyze multiple texts."""
    try:
        if (
            not settings().data.enable_term_extractor
            or NUMERICAL_SHORTER_ID
            or TextAnalyzer is None
        ):
            return {}

        analyzer = get_global_injector().get(TextAnalyzer)
        unique_terms = analyzer.get_unique_terms(
            texts,
            max_terms=max_terms,
            min_length=min_length,
            max_length=max_length,
            langs=langs,
        )
        return dict(enumerate(unique_terms))
    except Exception as e:
        logger.error(f"Failed to analyze texts: {e}")
        return {}


def analyze_texts_with_timeout(
    texts: list[str],
    min_length: int | None = SHORTER_ID_LENGTH,
    max_length: int | None = SHORTER_ID_LENGTH,
    max_terms: int = 1,
    langs: set[str] | None = None,
    timeout: float = 5.0,
) -> dict[int, list[str]]:
    def inner() -> dict[int, list[str]]:
        if (
            not settings().data.enable_term_extractor
            or NUMERICAL_SHORTER_ID
            or TextAnalyzer is None
        ):
            return {}
        try:
            analyzer = get_global_injector().get(TextAnalyzer)
            unique_terms = analyzer.get_unique_terms(
                texts,
                max_terms=max_terms,
                min_length=min_length,
                max_length=max_length,
                langs=langs,
            )
            return dict(enumerate(unique_terms))
        except Exception as e:
            logger.error(f"Failed to analyze texts: {e}")
            return {}

    with ThreadPoolExecutor() as executor:
        future = executor.submit(inner)
        try:
            return future.result(timeout=timeout)
        except Exception:
            logger.warning("Analyzing texts timed out or failed.")
            return {}


def generate_shorter_id(
    index: int, node_id: str, length: int = SHORTER_ID_LENGTH
) -> str:
    """Generate a shorter ID taking node ID as seed.

    Using this function, we can reduce token usage in
    the citation references and reduce the length of the citation references.
    """
    if NUMERICAL_SHORTER_ID:
        return f"{index:0{length}}"

    rng = random.Random(x=node_id)
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def init_nodes_with_shorter_ids(
    nodes: list[NodeWithScore], initial_index: int = 0
) -> list[NodeWithScore]:
    """Initialize nodes with shorter IDs.

    This function analyze the texts to try to get something contextually and unique.
    If nothing is retrieved, it generates a random shorter ID.
    This behavior is useful for generating shorter citation references and
    reducing the token use of the citation references and latency.
    """
    potential_shorted_ids = analyze_texts_with_timeout(
        [node.get_content(MetadataMode.LLM) for node in nodes],
        langs=set(settings().docling.langs),
        timeout=5.0,
    )
    for i, node in enumerate(nodes):
        if SHORTER_ID_FIELD not in node.metadata:
            related_terms = potential_shorted_ids.get(i)
            index = initial_index + i
            shorted_id = (
                related_terms[0]
                if related_terms
                else generate_shorter_id(index, node.node_id, SHORTER_ID_LENGTH)
            )
            node.metadata[SHORTER_ID_FIELD] = shorted_id.upper()
        if SHORTER_ID_FIELD not in node.node.excluded_llm_metadata_keys:
            node.node.excluded_llm_metadata_keys.append(SHORTER_ID_FIELD)
    return nodes


def exclude_metadata(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Exclude temporary metadata used for citation generation."""
    for node in nodes:
        MetadataHelper.exclude_metadata(node.node)
    return nodes


def convert_nodes_to_documents_list(
    nodes: list[NodeWithScore],
) -> list[Document]:
    """Convert nodes to documents list."""
    return [Document.from_node(node) for node in nodes or []]


def skip_return_nodes(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Skip return nodes."""
    return [n for n in nodes if MetadataFlags.SKIP_RETURN.value not in n.node.metadata]


def format_cite(i: int, doc: Document, index: int) -> str:
    """Format citation in the text."""
    data = {
        "id": doc.id,
        "index": index,
        "artifact_id": doc.document_id,
        "source_id": doc.id_,
        "correlation_id": doc.metadata.get(MetadataNode.CORRELATION_ID.value),
    }
    filtered_data = {k: v for k, v in data.items() if v is not None}
    attributes = "".join(f" {k}='{v}'" for k, v in filtered_data.items())
    return f"<citation{attributes}></citation>"


def _extract_citations_from_text(
    text: str | None,
) -> list[Citation]:
    """Extract citation markers and related document numbers from the text."""
    if not text:
        return []

    pattern = re.compile(
        r"<citation\s+([^>]+)>(.*?)</citation>",
        re.IGNORECASE,
    )
    cites = []
    for match in pattern.finditer(text):
        start_pos = match.start()
        end_pos = match.end()

        # Extract attributes
        attributes = match.group(1)
        attr_pattern = re.compile(r"(\w+)='(.*?)'")
        attr_dict = {
            attr_match.group(1): attr_match.group(2)
            for attr_match in attr_pattern.finditer(attributes)
        }

        # Extract element value
        elements = match.group(2)
        if not elements:
            elements = "{}"
        element_value = json.loads(elements) if elements else {}

        # Merge attributes and element value
        values = {**element_value, **attr_dict}

        # Add citation to the list
        cites.append(
            Citation(
                text=match.string[start_pos:end_pos],
                value=values,
                doc_id=values.get("id"),
                artifact_id=values.get("artifact_id"),
                source_id=values.get("source_id"),
            )
        )
    return cites




def extract_citations_by_original_text(
    text: str,
    documents: list[Document],
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
    split_token: str = DEFAULT_SPLIT_CITATION_TOKEN,
    shorter_id_length: int = SHORTER_ID_LENGTH,
    citation_indices: dict[str, int] | None = None,
    is_final: bool = False,
) -> tuple[str, list[Citation], dict[str, int]]:
    # Initialize an empty string to store the cleaned text
    citation_indices = citation_indices or {}

    result = ""
    start_len = len(start_token)
    end_len = len(end_token)

    # Model can generate brackets not normalized 【
    text = text.replace("【", start_token).replace("】", end_token)

    # Iterate through the text to remove malformed citations and save correct citations.
    # Backticks directly wrapping a citation are formatting noise and are not emitted.
    i = 0
    docs = []
    citation_placeholders: list[str] = []
    code_delimiter: str | None = None
    citation_wrapper_delimiter: str | None = None

    while i < len(text):
        if text[i] == "`":
            delimiter_end = i + 1
            while delimiter_end < len(text) and text[delimiter_end] == "`":
                delimiter_end += 1
            delimiter = text[i:delimiter_end]

            if citation_wrapper_delimiter == delimiter:
                citation_wrapper_delimiter = None
                i = delimiter_end
                continue

            if code_delimiter == delimiter:
                result += delimiter
                code_delimiter = None
                i = delimiter_end
                continue

            if delimiter_end == len(text):
                if is_final:
                    result += delimiter
                break

            if text[delimiter_end : delimiter_end + start_len] == start_token:
                citation_start = delimiter_end
                citation_end = citation_start + start_len
                while (
                    citation_end < len(text)
                    and text[citation_end : citation_end + end_len] != end_token
                ):
                    citation_end += 1

                if citation_end >= len(text):
                    break

                node_ids = [
                    node_id.strip()
                    for node_id in text[
                        citation_start + start_len : citation_end
                    ].split(split_token)
                ]
                valid_docs = [
                    doc
                    for node_id in node_ids
                    if (
                        doc := next(
                            (
                                document
                                for document in documents
                                if document.id.lower() == node_id.lower()
                            ),
                            None,
                        )
                    )
                ]
                if valid_docs:
                    placeholders = []
                    for doc in valid_docs:
                        placeholder_index = "".join(
                            f"n{digit}" for digit in str(len(docs))
                        )
                        placeholder = f"\ue000citation{placeholder_index}\ue001"
                        docs.append(doc)
                        citation_placeholders.append(placeholder)
                        placeholders.append(placeholder)
                    result += split_token.join(placeholders)
                    i = citation_end + end_len
                    if text[i : i + len(delimiter)] == delimiter:
                        i += len(delimiter)
                    else:
                        citation_wrapper_delimiter = delimiter
                    continue

            result += delimiter
            code_delimiter = delimiter
            i = delimiter_end
        elif text[i : i + start_len] == start_token:
            # Check if we have a complete citation
            j = i + start_len
            while j < len(text) and text[j : j + end_len] != end_token:
                j += 1

            if j < len(text) and text[j : j + end_len] == end_token:
                node_ids = [
                    id.strip() for id in text[i + start_len : j].split(split_token)
                ]
                valid_docs = []
                for node_id in node_ids:
                    doc = next(
                        (doc for doc in documents if doc.id.lower() == node_id.lower()),
                        None,
                    )
                    if doc:
                        valid_docs.append(doc)
                if valid_docs:
                    placeholders = []
                    for doc in valid_docs:
                        placeholder_index = "".join(
                            f"n{digit}" for digit in str(len(docs))
                        )
                        placeholder = f"\ue000citation{placeholder_index}\ue001"
                        docs.append(doc)
                        citation_placeholders.append(placeholder)
                        placeholders.append(placeholder)
                    result += split_token.join(placeholders)
                    i = j + end_len
                else:
                    # No valid docs in citation, treat as regular text
                    result += text[i : j + end_len]
                    i = j + end_len
            else:
                # Incomplete citation detected: drop the
                # citation content and stop processing further.
                # This change fixes the issue by not
                # appending any incomplete citation text.
                i = len(text)
                continue
        else:
            result += text[i]
            i += 1

    # Process citations
    pattern = re.compile(
        rf"{re.escape(start_token)}?[A-Z0-9]{shorter_id_length}{re.escape(end_token)}?"
    )
    for match in pattern.finditer(result):
        # If citation is well-formed, skip
        word = match.group(0) if match.groups() else ""
        if not word or (word.startswith(start_token) and word.endswith(end_token)):
            continue

        # Try to find the related document, if no document found, skip
        doc = next((doc for doc in documents if doc.id in word), None)
        if not doc:
            continue

        # Remove doc reference
        new_token = word.replace(doc.id, "", 1).lstrip().rstrip()
        result = result[: match.start()] + new_token + result[match.end() :]

    # Replace citations with sequential numbers, just first occurrence
    max_index = max(citation_indices.values(), default=-1)
    current_index = max_index + 1
    processed_docs = []
    for i, (doc, placeholder) in enumerate(
        zip(docs, citation_placeholders, strict=True)
    ):
        if doc.id_ not in processed_docs:
            processed_docs.append(doc.id_)

        if doc.id_ in citation_indices:
            # If we already have this document, use the existing index
            index = citation_indices[doc.id_]
        else:
            # Otherwise, assign a new index
            index = current_index
            current_index += 1
        citation_indices[doc.id_] = index

        citation = format_cite(i, doc, index)
        result = result.replace(placeholder, citation, 1)

    return result, _extract_citations_from_text(result), citation_indices


async def deduplicate_documents_in_history(
    chat_history: list[ChatMessage] | None,
    prompt_builder_service: Any | None = None,
) -> list[ChatMessage] | None:
    if not chat_history:
        return chat_history

    last_seen: dict[str, int] = {}
    for i, msg in enumerate(chat_history):
        for doc in await extract_sources_from_history([msg]):
            last_seen[doc.id_] = i

    if not last_seen:
        return chat_history

    if prompt_builder_service is None:
        from private_gpt.components.prompts.prompt_builder import PromptBuilderService

        prompt_builder_service = get_global_injector().get(PromptBuilderService)

    for i, msg in enumerate(chat_history):
        for key in ("source",):
            blocks = msg.additional_kwargs.get(key)
            if not isinstance(blocks, list):
                continue

            has_documents = False
            for block in blocks:
                if isinstance(block, SourceBlock):
                    has_documents = has_documents or bool(block.sources)
                    block.sources = [
                        s
                        for s in block.sources
                        if last_seen.get(Document.from_source(s).id_) == i
                    ]

            documents = [
                Document.from_source(source)
                for content_block in blocks
                if isinstance(content_block, SourceBlock)
                for source in content_block.sources
            ]
            if documents:
                prompt, _ = prompt_builder_service.create_context_prompt(
                    documents=documents,
                    generate_citations=True,
                )
                msg.blocks = [TextBlock(text=prompt.format())]
                msg.additional_kwargs[key] = [
                    b for b in blocks if not isinstance(b, SourceBlock) or b.sources
                ]
            elif not documents and has_documents:
                msg.blocks = [
                    TextBlock(
                        text=(
                            "The documents requested by this tool call were already retrieved "
                            "in a later message and have been intentionally removed to avoid duplication. "
                            "This is not an error — do not retry this tool call. "
                            "Use only the sources available in subsequent messages."
                        )
                    )
                ]
                msg.additional_kwargs[key] = []

    return chat_history


async def extract_sources_from_history(
    chat_history: list[ChatMessage] | None,
) -> list[Document]:
    chat_history = chat_history or []
    documents: list[Document] = []

    # Extract from sources from the chat history
    source_messages: list[SourceType] = [
        source
        for message in chat_history
        if message.additional_kwargs.get("source")
        and isinstance(message.additional_kwargs["source"], list)
        for block in message.additional_kwargs["source"]
        if isinstance(block, SourceBlock)
        for source in block.sources
    ]
    documents.extend([Document.from_source(source) for source in source_messages])

    # Deduplicate documents
    documents = list({doc.id_: doc for doc in documents}.values())

    # TODO: Remove in the future
    # Since FE doesn't support citations of webpage types,
    # we need to discard them for now
    documents = [doc for doc in documents if doc.type != "webpage"]

    # If we are using numerical shorter IDs, we need to generate them
    if NUMERICAL_SHORTER_ID:
        unique_shorter_ids = {
            doc.metadata.get(MetadataFlags.SHORTER_ID.value) for doc in documents
        }
        if len(unique_shorter_ids) != len(documents):
            for i, doc in enumerate(documents):
                doc.shorter_id = generate_shorter_id(i, doc.id_, SHORTER_ID_LENGTH)

    return documents


async def extract_citations_from_history(
    chat_history: list[ChatMessage],
) -> list[Citation]:
    """Extract citations from the chat history.

    This function extracts citations from the chat history
    and returns them as a list of Citation objects.
    """
    if not chat_history:
        return []

    citations = []
    for message in [
        m for m in chat_history if m.content and m.role == MessageRole.ASSISTANT
    ]:
        # Extract citations from the message content
        extracted_citations = _extract_citations_from_text(message.content)
        citations.extend(extracted_citations)

        # Extract citations from the thinking block if it exists
        if "thinking" in message.additional_kwargs:
            thinking_blocks: list[ThinkingBlock] = message.additional_kwargs["thinking"]
            for thinking_block in thinking_blocks:
                if not isinstance(thinking_block, ThinkingBlock):
                    continue
                extracted_citations.extend(
                    _extract_citations_from_text(thinking_block.thinking or "")
                )

    return citations


async def process_history_citations(
    chat_history: list[ChatMessage],
    correlation_id: str | None = None,
    **kwargs: Any,
) -> tuple[list[ChatMessage], list[Document], list[Citation]]:
    if not chat_history:
        return chat_history, [], []

    # Extract documents from the chat history
    documents_list: list[Document] = await extract_sources_from_history(
        chat_history,
    )

    # Init correlation ID for documents if available
    documents_list = init_documents_with_correlation_id(
        documents=documents_list, correlation_id=correlation_id, **kwargs
    )

    # Extract citations from the last user block
    user_blocks = get_user_blocks(chat_history)
    current_citations = await extract_citations_from_history(
        user_blocks[-1] if user_blocks else []
    )

    # Remove any non-known documents
    chat_history = replace_citations_in_text(chat_history, documents_list)

    return chat_history, documents_list, current_citations


def replace_citations_in_text(
    chat_history: list[ChatMessage],
    documents: list[Document],
    unk_token: str = DEFAULT_UNK_TOKEN,
    start_token: str = ORIGINAL_START_TOKEN,
    end_token: str = ORIGINAL_END_TOKEN,
) -> list[ChatMessage]:
    """Replace citations in the chat history with the document IDs.

    When the chat history contains citations, this function replaces the citations
    with the document IDs. If the document ID is not found in the documents list,
    it is replaced with the UNK token. The UNK token is used to represent unknown
    document IDs in the chat history. This function prevents the model from
    generating citations for unknown document IDs.

    If we are using numerical shorter IDs, we cannot keep identity and trace
    so we use the UNK token to prevent the model from generating citations
    from older IDs.
    """
    for message in [
        m for m in chat_history if m.content and m.role == MessageRole.ASSISTANT
    ]:
        citations = _extract_citations_from_text(message.content)
        for citation in citations:
            citation_id = (
                citation.doc_id
                if not NUMERICAL_SHORTER_ID
                and any(doc.id == citation.doc_id for doc in documents)
                else unk_token
            )
            new_text = f"{start_token}{citation_id}{end_token}"
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    block.text = block.text.replace(
                        citation.text,
                        new_text if citation_id != unk_token else "",
                    )

        if "thinking" in message.additional_kwargs:
            thinking_blocks: list[ThinkingBlock] = message.additional_kwargs["thinking"]
            for thinking_block in thinking_blocks:
                if not isinstance(thinking_block, ThinkingBlock):
                    continue
                citations = _extract_citations_from_text(thinking_block.thinking or "")
                for citation in citations:
                    citation_id = (
                        citation.doc_id
                        if not NUMERICAL_SHORTER_ID
                        and any(doc.id == citation.doc_id for doc in documents)
                        else unk_token
                    )
                    new_text = f"{start_token}{citation_id}{end_token}"
                    thinking_block.thinking = thinking_block.thinking.replace(
                        citation.text,
                        new_text if citation_id != unk_token else "",
                    )

    return chat_history


def init_documents_with_correlation_id(
    documents: list[Document],
    correlation_id: str | None = None,
    **kwargs: Any,
) -> list[Document]:
    """Initialize documents with correlation ID."""
    if not documents or not correlation_id:
        return documents

    for doc in documents:
        doc.update_metadata(
            key=MetadataNode.CORRELATION_ID.value,
            value=correlation_id,
        )

    return documents
