from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from llama_index.core.schema import MetadataMode
from pydantic import BaseModel, Field

from private_gpt.components.ingest.utils import get_extension, get_file_name
from private_gpt.components.readers.factories.factory import ReaderFactoryRegistry
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode
from private_gpt.components.readers.registry import ReaderRegistry
from private_gpt.server.content.content_router import ContentFormat, ContentTree
from private_gpt.server.ingest.convert_service import ConvertService
from private_gpt.server.utils.artifact_input import IngestableArtifactType
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.utils.openapi_models import OpenAPIValidationErrorResponse

convert_router = APIRouter(
    prefix="/v1/artifacts",
    dependencies=[Depends(authenticated)],
    tags=["Artifacts"],
    responses={401: {"description": "Unauthorized"}},
)


class ConvertBody(BaseModel):
    input: IngestableArtifactType = Field(
        ..., description="File content as base64, remote URI, or plain text"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata, must include 'file_name' to resolve file extension",
        examples=[
            {"file_name": "report.pdf"},
            {"file_name": "document.docx", "author": "John Doe"},
        ],
    )
    reader: str | None = Field(
        default=None,
        description="Reader to use for parsing. If omitted the default reader for the file type is used.",
        examples=["markitdown", "docling"],
    )
    format: ContentFormat = Field(
        default=ContentFormat.Markdown,
        description="Output format: 'markdown' returns text, 'object' returns a content tree.",
        examples=["markdown", "object"],
    )
    execute_transformations: bool = Field(
        default=False,
        description=(
            "Whether to run the transformation pipeline (chunking, tokenization, "
            "tree building, etc.) on the parsed content. Defaults to False because "
            "this endpoint is meant as a lightweight preview/conversion of a file, "
            "and transformations are costly and only needed to obtain a fully "
            "structured 'object' content tree. Set to True to get a chunked tree "
            "when using format='object'; with format='markdown' this has no "
            "practical effect since only the flat parsed text is returned."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": {
                        "type": "text",
                        "value": "# Hello\n\nThis is a simple markdown document.",
                    },
                    "metadata": {"file_name": "hello.md"},
                    "format": "markdown",
                },
                {
                    "input": {"type": "file", "value": "JVBERi0xLjQKJaqrrK0KMS..."},
                    "metadata": {"file_name": "report.pdf"},
                    "reader": "markitdown",
                    "format": "markdown",
                },
                {
                    "input": {
                        "type": "uri",
                        "value": "https://example.com/document.pdf",
                    },
                    "metadata": {"file_name": "document.pdf"},
                    "format": "object",
                },
            ]
        }
    }


class ConvertResponse(BaseModel):
    content: str | ContentTree = Field(
        ..., description="Parsed file content in the requested format"
    )
    reader: str = Field(..., description="Reader that was used to parse the file")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "# Annual Report 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period...",
                    "reader": "markitdown",
                },
                {
                    "content": {
                        "id": "root",
                        "type": "document",
                        "content": "",
                        "children": [
                            {
                                "id": "section-1",
                                "type": "SectionNode",
                                "content": "Annual Report 2023",
                                "children": [
                                    {
                                        "id": "text-1",
                                        "type": "TextNode",
                                        "content": "Executive Summary\n\nFiscal year 2023 marked a transformative period...",
                                        "children": [],
                                    }
                                ],
                            }
                        ],
                    },
                    "reader": "docling",
                },
            ]
        }
    }


class ReaderInfo(BaseModel):
    extensions: list[str] = Field(
        ..., description="File extensions this reader can process"
    )


class ReadersResponse(BaseModel):
    data: dict[str, ReaderInfo] = Field(
        ..., description="Available readers and the extensions each one supports"
    )


@convert_router.get(
    "/readers",
    response_model=ReadersResponse,
    summary="List Available Readers",
    description="Returns all registered readers and the file extensions each one supports.",
)
def list_readers(request: Request) -> ReadersResponse:
    registry: ReaderRegistry = request.state.injector.get(ReaderRegistry)
    factory_registry: ReaderFactoryRegistry = request.state.injector.get(
        ReaderFactoryRegistry
    )

    # Invert the registry: extension → [readers] becomes reader → [extensions]
    reader_extensions: dict[str, list[str]] = {}
    for extension in sorted(registry.get_all_extensions()):
        for name in registry.get_reader_names(extension):
            try:
                factory_registry.get_factory(name)  # validate the factory exists
            except ValueError:
                continue
            reader_extensions.setdefault(name, []).append(extension)
    return ReadersResponse(
        data={
            name: ReaderInfo(extensions=exts)
            for name, exts in reader_extensions.items()
        }
    )


@convert_router.post(
    "/convert",
    response_model=ConvertResponse,
    summary="Convert File to Markdown or Tree",
    description=(
        "Parse a file using the document readers and return its content as markdown text "
        "or a structured content tree, without ingesting it into the knowledge base."
    ),
    responses={
        200: {
            "description": "Successful conversion",
            "content": {
                "application/json": {
                    "examples": {
                        "markdown_result": {
                            "summary": "Converted content as markdown text",
                            "value": {
                                "content": "# Annual Report 2023\n\nExecutive Summary\n\nFiscal year 2023 marked a transformative period...",
                                "reader": "markitdown",
                            },
                        },
                        "object_result": {
                            "summary": "Converted content as structured tree",
                            "value": {
                                "content": {
                                    "id": "root",
                                    "type": "document",
                                    "content": "",
                                    "children": [
                                        {
                                            "id": "section-1",
                                            "type": "SectionNode",
                                            "content": "Annual Report 2023",
                                            "children": [
                                                {
                                                    "id": "text-1",
                                                    "type": "TextNode",
                                                    "content": "Executive Summary\n\nFiscal year 2023 marked a transformative period...",
                                                    "children": [],
                                                }
                                            ],
                                        }
                                    ],
                                },
                                "reader": "docling",
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": OpenAPIValidationErrorResponse,
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_reader": {
                            "summary": "Reader not supported for file extension",
                            "value": {
                                "detail": "Reader 'docling' is not supported for '.txt'. Valid readers: ['text']"
                            },
                        },
                        "invalid_base64": {
                            "summary": "Invalid base64 encoded file",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "input", "value"],
                                        "msg": "File input requires valid base64 encoded content",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                    }
                }
            },
        },
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ConvertBody"},
                    "examples": {
                        "text": {
                            "summary": "Plain text content",
                            "value": {
                                "input": {
                                    "type": "text",
                                    "value": "# Hello\n\nThis is a simple markdown document.",
                                },
                                "metadata": {"file_name": "hello.md"},
                                "format": "markdown",
                            },
                        },
                        "file_base64": {
                            "summary": "File content (base64)",
                            "value": {
                                "input": {
                                    "type": "file",
                                    "value": "JVBERi0xLjQKJaqrrK0KMS...",
                                },
                                "metadata": {"file_name": "report.pdf"},
                                "reader": "markitdown",
                                "format": "markdown",
                            },
                        },
                        "uri": {
                            "summary": "Remote URI",
                            "value": {
                                "input": {
                                    "type": "uri",
                                    "value": "https://example.com/document.pdf",
                                },
                                "metadata": {"file_name": "document.pdf"},
                                "format": "object",
                            },
                        },
                        "object_format": {
                            "summary": "Return structured content tree",
                            "value": {
                                "input": {
                                    "type": "text",
                                    "value": "# Section\n\nSome content here.",
                                },
                                "metadata": {"file_name": "doc.md"},
                                "format": "object",
                            },
                        },
                    },
                }
            },
            "required": True,
            "description": (
                "Request body for converting a file to markdown or structured tree.\n\n"
                "Input Types:\n"
                "* file: Base64-encoded file content — set 'file_name' in metadata to specify the extension\n"
                "* uri: Remote URL or S3 URI pointing to the file\n"
                "* text: Plain text or markdown content\n\n"
                "Format Options:\n"
                "* markdown (default): Returns parsed content as a flat markdown string\n"
                "* object: Returns a hierarchical content tree with typed nodes\n\n"
                "Transformations:\n"
                "* 'execute_transformations' defaults to False - the transformation "
                "pipeline is skipped since this endpoint is meant as a lightweight "
                "preview/conversion, not ingestion\n"
                "* Set it to True to get a fully chunked content tree when using "
                "format='object'\n\n"
                "Reader Selection:\n"
                "* Omit 'reader' to use the default reader for the detected file type\n"
                "* Use GET /v1/artifacts/readers to list available readers and their supported extensions"
            ),
        }
    },
)
def convert_content(request: Request, body: ConvertBody) -> ConvertResponse:
    service: ConvertService = request.state.injector.get(ConvertService)
    registry: ReaderRegistry = request.state.injector.get(ReaderRegistry)

    input_content = body.input.to_binary_content(get_file_name(body.metadata))
    extension = get_extension(input_content.filename)

    if body.reader:
        valid_readers = registry.get_reader_names(extension)
        if body.reader not in valid_readers:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Reader '{body.reader}' is not supported for '{extension}'. "
                    f"Valid readers: {valid_readers}"
                ),
            )

    with service.temporary_file(
        lambda: service.data_path_from_bin_data(
            input_content.data, get_extension(input_content.filename)
        )
    ) as data_path:
        result = service.convert_file(
            file_data=data_path,
            file_metadata={
                **(body.metadata or {}),
                "file_name": input_content.filename,
            },
            reader=body.reader,
            # Transformations (chunking, tokenization, tree building) are only
            # run when explicitly requested via `body.execute_transformations`
            # (default False) - see ConvertBody.execute_transformations for why.
            execute_transformations=body.execute_transformations,
        )

    metadata_mode = (
        TreeMetadataMode.USER
        if body.format == ContentFormat.Markdown
        else TreeMetadataMode.NONE
    )
    generic_metadata_mode = (
        MetadataMode.ALL if body.format == ContentFormat.Markdown else MetadataMode.NONE
    )
    roots = [n for n in result.nodes if isinstance(n, TreeNode) and n.parent is None]

    if body.format == ContentFormat.Markdown:
        if roots:
            content: str | ContentTree = ContentTree.from_node(
                roots[0], mode=metadata_mode
            ).content
        else:
            content = "\n\n".join(
                text
                for n in result.nodes
                if (text := n.get_content(metadata_mode=generic_metadata_mode))
            )
    elif roots:
        content = ContentTree.from_node(roots[0], mode=metadata_mode)
    else:
        content = ContentTree(
            id="root",
            type="document",
            content="",
            children=[
                ContentTree(
                    id=n.id_,
                    type=n.get_type() if hasattr(n, "get_type") else "text",
                    content=n.get_content(metadata_mode=generic_metadata_mode),
                    children=[],
                )
                for n in result.nodes
            ],
        )

    return ConvertResponse(content=content, reader=result.reader)
