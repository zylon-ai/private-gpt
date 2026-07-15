from fastapi import APIRouter, Depends, Request, Response
from starlette.responses import StreamingResponse

from private_gpt.chat.input_models import (
    CountTokensInput,
    CountTokensOutput,
    System,
    Thinking,
    ToolChoice,
)
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.events.models import (
    FatalError,
    Message,
)
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat.chat_service import ChatService, ChatValidationResult
from private_gpt.server.chat_async.chat_async_facade import ChatAsyncFacadeService
from private_gpt.server.utils.auth import authenticated

chat_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(authenticated)],
    tags=["Messages"],
    responses={401: {"description": "Unauthorized"}},
)


@chat_router.post(
    "/messages",
    response_model=None,
    summary="Messages",
    responses={
        200: {
            "model": Message,
            "description": "Successful chat message",
            "content": {
                "application/json": {
                    "examples": {
                        "text_response": {
                            "summary": "Standard text response",
                            "value": {
                                "id": "msg_12345",
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "start_timestamp": "2025-07-10T09:11:16.003615Z",
                                        "stop_timestamp": "2025-07-10T09:11:28.048942Z",
                                        "text": "Based on the analysis, I found 3 potential security issues...",
                                    }
                                ],
                                "model": "private-gpt",
                                "stop_reason": "end_turn",
                                "stop_sequence": None,
                                "usage": {"input_tokens": 432, "output_tokens": 89},
                            },
                        },
                        "tool_use_response": {
                            "summary": "Response with tool usage",
                            "value": {
                                "id": "msg_67890",
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "I'll help you with precise temperature control.",
                                    },
                                    {
                                        "type": "tool_use",
                                        "id": "scan_001",
                                        "name": "security_scanner",
                                        "input": {
                                            "directory": "/src",
                                            "scan_type": "vulnerability",
                                        },
                                    },
                                ],
                                "model": "private-gpt",
                                "stop_reason": "tool_use",
                                "usage": {"input_tokens": 156, "output_tokens": 234},
                            },
                        },
                        "tool_use_and_result_response": {
                            "summary": "Response with tool usage and result",
                            "value": {
                                "id": "msg_67890",
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "I'll help you with precise temperature control.",
                                    },
                                    {
                                        "type": "tool_use",
                                        "id": "scan_001",
                                        "name": "security_scanner",
                                        "input": {
                                            "directory": "/src",
                                            "scan_type": "vulnerability",
                                        },
                                    },
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "scan_001",
                                        "content": [
                                            {
                                                "type": "text",
                                                "start_timestamp": "2025-07-10T09:11:40.123456Z",
                                                "stop_timestamp": "2025-07-10T09:11:45.654321Z",
                                                "text": "Scan complete. Found 3 vulnerabilities in /src.",
                                            }
                                        ],
                                    },
                                    {
                                        "type": "text",
                                        "text": "Based on the scan results, you should address the vulnerabilities found in /src...",
                                    },
                                ],
                                "model": "private-gpt",
                                "stop_reason": "end_turn",
                                "usage": {"input_tokens": 156, "output_tokens": 324},
                            },
                        },
                    }
                },
                "text/event-stream": {
                    "examples": {
                        "streaming_response": {
                            "summary": "Server-sent events for streaming",
                            "value": (
                                "event: message_start\n"
                                'data: {"type":"message_start","message":{"id":"msg_12345","type":"message","role":"assistant","content":[],"model":"private-gpt","usage":{}}}\n\n'
                                "event: content_block_start\n"
                                'data: {"type":"content_block_start","block_id":"block_001","content_block":{"type":"text","start_timestamp":"2025-07-10T09:12:41.819953Z","text":""}}\n\n'
                                "event: content_block_delta\n"
                                'data: {"type":"content_block_delta","block_id":"block_001","delta":{"type":"text_delta","text":"Analyzing your codebase"}}\n\n'
                                "event: content_block_stop\n"
                                'data: {"type":"content_block_stop","stop_timestamp":"2025-07-10T09:12:54.044672Z","block_id":"block_001"}\n\n'
                                "event: message_delta\n"
                                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":432,"output_tokens":89}}\n\n'
                                "event: message_stop\n"
                                'data: {"type":"message_stop"}\n\n'
                            ),
                        }
                    }
                },
            },
        },
        422: {
            "description": "Validation Error - Invalid request parameters",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_messages": {
                            "summary": "Empty messages array",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "messages"],
                                        "msg": "Messages cannot be empty",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                        "invalid_tool_choice": {
                            "summary": "Invalid tool choice",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "tool_choice"],
                                        "msg": "Tool choice 'nonexistent_tool' is not in the provided tools",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        },
                        "json_schema_with_tools": {
                            "summary": "JSON schema incompatible with tools",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body"],
                                        "msg": "Tools are not supported when response_format is set to json_schema",
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
    tags=["Messages"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for chat completion supporting multi-turn "
                "conversations with AI models.\n\n"
                "Contains message history, tool definitions, system prompts, "
                "and response configuration. Supports both streaming and "
                "non-streaming responses with optional tool usage, citations, "
                "and advanced sampling parameters.\n\n"
                "The request body defines the complete conversation context "
                "and AI behavior parameters for generating responses."
            ),
        }
    },
)
async def chat_messages(
    request: Request, body: ChatBody
) -> Message | FatalError | StreamingResponse:
    """Generate a chat completion from a conversation history.

    This endpoint enables multi-turn conversations with the AI model, with
    optional tool support and comprehensive message validation.

    Key Features:
    * Multi-turn conversations: Support for system, user, and assistant
      messages
    * Tool Support: Full tool use/result validation with automatic or manual
      selection
    * Citations: Enable `system.citations.enabled` to include references in
      responses
    * Streaming: Enable `stream` for partial updates in real-time
    * Default Prompts: Enable `system.use_default_prompt` for using Zylon
      prompts
    * Thinking: Enable `thinking.enabled` for step-by-step reasoning
      capabilities
    * Sampling Parameters: Control randomness with temperature, top_p,
      top_k, etc.

    Notes:
    * Tool use/result blocks must be properly paired within assistant
      messages
    * Tool choice type must be 'auto', 'tool', or 'none'
    * When tool_choice.type is 'tool', tool_choice.name must specify a
      valid tool
    * All message content is validated for completeness and proper structure
    * Last message must be from user or assistant for proper conversation
      flow
    * MCP servers provide external tool capabilities via Model Context
      Protocol
    * Sampling parameters control response randomness and token selection
    """
    chat_facade_service: ChatAsyncFacadeService = request.state.injector.get(
        ChatAsyncFacadeService
    )

    return await chat_facade_service.chat(
        request,
        body,
    )


@chat_router.post(
    "/messages/count_tokens",
    response_model=CountTokensOutput,
    summary="Count tokens in a Message",
    tags=["Messages"],
)
async def count_message_tokens(
    request: Request,
    body: CountTokensInput,
) -> CountTokensOutput:
    chat_service: ChatService = request.state.injector.get(ChatService)
    request_mapper: ChatRequestMapper = request.state.injector.get(ChatRequestMapper)
    system_value: list[System] = body.system or [System()]

    chat_body = ChatBody(
        model=body.model,
        stream=False,
        messages=body.messages,
        tools=body.tools,
        tool_choice=body.tool_choice or ToolChoice(),
        system=system_value,
        thinking=body.thinking or Thinking(),
        output_config=body.output_config,
        cache_control=body.cache_control,
    )
    chat_request = await request_mapper.create_request_from_body(chat_body)
    return await chat_service.count_tokens(chat_request)


@chat_router.post(
    "/messages/validate",
    response_model=ChatValidationResult,
    summary="Validate Messages Request",
    responses={
        200: {
            "description": "Validation completed",
            "content": {
                "application/json": {
                    "examples": {
                        "valid_request": {
                            "summary": "Valid request example",
                            "value": {
                                "valid": True,
                                "errors": [],
                                "warnings": [],
                                "request_summary": {
                                    "message_count": 3,
                                    "has_tools": True,
                                    "stream_enabled": False,
                                    "tool_choice": "auto",
                                },
                            },
                        }
                    }
                }
            },
        },
        400: {
            "description": "Validation completed",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_request": {
                            "summary": "Invalid request example",
                            "value": {
                                "valid": False,
                                "errors": [
                                    "Messages cannot be empty",
                                    "Tool choice 'nonexistent_tool' is not in the provided tools",
                                ],
                            },
                        },
                    }
                }
            },
        },
        422: {
            "description": "Request body validation error",
            "content": {
                "application/json": {
                    "examples": {
                        "malformed_request": {
                            "summary": "Malformed request body",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "messages"],
                                        "msg": "field required",
                                        "type": "value_error.missing",
                                    }
                                ]
                            },
                        }
                    }
                }
            },
        },
    },
    tags=["Messages"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for chat completion supporting multi-turn "
                "conversations with AI models.\n\n"
                "Contains message history, tool definitions, system prompts, "
                "and response configuration. Supports both streaming and "
                "non-streaming responses with optional tool usage, citations, "
                "and advanced sampling parameters.\n\n"
                "The request body defines the complete conversation context "
                "and AI behavior parameters for generating responses."
            ),
        }
    },
)
async def validate_messages(
    request: Request, response: Response, body: ChatBody
) -> ChatValidationResult:
    """Validate a chat completion request without executing it.

    This endpoint performs a dry-run validation of the chat request,
    checking for:

    * Message structure and content validation
    * Tool definitions and tool_choice compatibility
    * Parameter ranges and combinations
    * Conversation flow and message ordering
    * Response format compatibility with other options

    Returns detailed validation results including errors. Use this endpoint to
    validate requests before sending them to the main chat endpoint.

    Notes:
    * No tokens are consumed during validation
    * All validation rules match the main /messages endpoint
    * Warnings indicate potential issues but don't prevent execution
    * Request summary provides insights into the parsed request structure
    """
    chat_service: ChatService = request.state.injector.get(ChatService)
    request_mapper: ChatRequestMapper = request.state.injector.get(ChatRequestMapper)

    chat_request = await request_mapper.create_request_from_body(body)
    chat_request.tool_config.validation_mode = (  # Force eager validation
        ToolValidationMode.EAGER
    )

    result = await chat_service.validate(chat_request)
    if not result.valid:
        response.status_code = 400
    return result
