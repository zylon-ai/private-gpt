from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from private_gpt.components.streaming.providers.models import (
    StreamStatus,
)
from private_gpt.events.utils import to_sse_stream
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService
from private_gpt.server.utils.auth import authenticated


class StreamMetadata(BaseModel):
    """Stream metadata and status information for asynchronous chat completions."""

    model_config = {
        "json_schema_extra": {
            "description": "Stream metadata and status information for asynchronous chat completions",
            "examples": [
                {
                    "message_id": "msg_async_12345",
                    "status": "pending",
                    "created_at": "2025-07-10T09:11:16.003615Z",
                    "updated_at": "2025-07-10T09:11:16.003615Z",
                    "completed_at": None,
                    "error_message": None,
                    "stream_type": "default",
                    "metadata": {},
                },
                {
                    "message_id": "msg_async_67890",
                    "status": "processing",
                    "created_at": "2025-07-10T09:11:16.003615Z",
                    "updated_at": "2025-07-10T09:11:20.123456Z",
                    "completed_at": None,
                    "error_message": None,
                    "stream_type": "default",
                    "metadata": {
                        "tokens_processed": 156,
                        "tools_used": ["security_scanner"],
                    },
                },
            ],
        }
    }

    message_id: str = Field(
        description="Unique identifier for the stream",
    )
    status: StreamStatus = Field(
        description="Current status of the stream",
    )
    created_at: datetime = Field(
        description="Timestamp when the stream was created",
    )
    updated_at: datetime = Field(
        description="Timestamp when the stream was last updated",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the stream was completed, if applicable",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the stream encountered an error",
    )
    stream_type: str = Field(
        default="default",
        description="Type of the stream, used for categorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the stream",
    )


class ChatResponse(BaseModel):
    """Response model for initiated asynchronous chat completion streams."""

    model_config = {
        "json_schema_extra": {
            "description": "Response model for initiated asynchronous chat completion streams",
            "examples": [
                {
                    "message_id": "msg_async_12345",
                    "status": "pending",
                    "message": "Request initiated successfully",
                },
                {
                    "message_id": "custom_msg_67890",
                    "status": "pending",
                    "message": "Request initiated successfully",
                },
            ],
        }
    }

    message_id: str = Field(description="Unique identifier for the initiated stream")
    status: StreamStatus = Field(
        description="Initial status of the stream (typically 'pending')"
    )
    message: str = Field(
        default="Request initiated successfully",
        description="Confirmation message for successful stream initiation",
    )


class ChatCancellationResponse(BaseModel):
    """Response model for cancelled asynchronous chat streams."""

    model_config = {
        "json_schema_extra": {
            "description": "Response model for cancelled asynchronous chat streams",
            "examples": [
                {
                    "message": "Stream cancelled successfully",
                    "message_id": "msg_async_12345",
                },
                {
                    "message": "Stream cancelled successfully",
                    "message_id": "msg_async_long_running",
                },
            ],
        }
    }

    message: str = Field(
        description="Confirmation message for successful stream cancellation"
    )
    message_id: str = Field(description="Unique identifier of the cancelled stream")


chat_router = APIRouter(
    prefix="/v1/messages/async",
    dependencies=[Depends(authenticated)],
    tags=["Async Messages"],
    responses={401: {"description": "Unauthorized"}},
)


@chat_router.post(
    path="",
    response_model=ChatResponse,
    summary="Initiate Async Chat Stream",
    responses={
        200: {
            "model": ChatResponse,
            "description": "Chat stream initiated successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "stream_initiated": {
                            "summary": "Stream initiated successfully",
                            "value": {
                                "message_id": "msg_async_12345",
                                "status": "pending",
                                "message": "Request initiated successfully",
                            },
                        },
                        "stream_with_custom_id": {
                            "summary": "Stream initiated with custom message_id",
                            "value": {
                                "message_id": "custom_msg_67890",
                                "status": "pending",
                                "message": "Request initiated successfully",
                            },
                        },
                    }
                }
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
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatBody"}
                }
            },
            "required": True,
            "description": (
                "Request body for initiating asynchronous chat completion "
                "with multi-turn conversation support.\n\n"
                "Contains message history, tool definitions, system prompts, "
                "and response configuration. The request initiates a background "
                "process that streams events, which can be observed "
                "via the stream endpoint.\n\n"
                "The request body defines the complete conversation context "
                "and AI behavior parameters for generating asynchronous responses."
            ),
        },
    },
)
async def chat_messages(
    request: Request,
    body: ChatBody,
    message_id: Annotated[
        str | None,
        Query(
            description=(
                "Optional custom identifier for the stream. "
                "If not provided, a unique ID will be generated automatically."
            ),
            examples=["custom_msg_12345"],
        ),
    ] = None,
) -> ChatResponse:
    """Initiate an asynchronous chat completion stream.

    This endpoint starts an asynchronous chat completion process that streams
    events. Unlike synchronous chat, this endpoint returns immediately
    with a message_id that can be used to observe the stream progress.

    Key Features:
    * Asynchronous Processing: Non-blocking request handling with immediate response
    * Stream Observation: Use returned message_id to observe real-time events
    * Works exactly like synchronous chat, but in an async manner

    Notes:
    * Optional message_id query parameter for custom stream identification
    * Stream events follow the same format as synchronous chat responses
    * Stream status can be monitored via status endpoint
    """
    chat_service: ChatAsyncService = request.state.injector.get(ChatAsyncService)
    chat_request_mapper: ChatRequestMapper = request.state.injector.get(
        ChatRequestMapper
    )
    message_id = await chat_service.initiate_chat_stream(
        await chat_request_mapper.create_request_from_body(body), message_id=message_id
    )

    return ChatResponse(
        message_id=message_id,
        status=StreamStatus.PENDING,
        message="Request initiated successfully",
    )


@chat_router.get(
    "/{message_id}/stream",
    summary="Observe Async Chat Stream Events",
    responses={
        200: {
            "description": "Server-sent events stream with chat completion data",
            "content": {
                "text/event-stream": {
                    "examples": {
                        "async_streaming_response": {
                            "summary": "Server-sent events for async streaming",
                            "value": (
                                "event: message_start\n"
                                'data: {"type":"message_start","message":{"id":"msg_async_12345","type":"message","role":"assistant","content":[],"model":"private-gpt","usage":{}}}\n\n'
                                "event: content_block_start\n"
                                'data: {"type":"content_block_start","block_id":"block_001","content_block":{"type":"text","start_timestamp":"2025-07-10T09:12:41.819953Z","text":""}}\n\n'
                                "event: content_block_delta\n"
                                'data: {"type":"content_block_delta","block_id":"block_001","delta":{"type":"text_delta","text":"Processing your request asynchronously"}}\n\n'
                                "event: content_block_stop\n"
                                'data: {"type":"content_block_stop","stop_timestamp":"2025-07-10T09:12:54.044672Z","block_id":"block_001"}\n\n'
                                "event: message_delta\n"
                                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":432,"output_tokens":89}}\n\n'
                                "event: message_stop\n"
                                'data: {"type":"message_stop"}\n\n'
                            ),
                        },
                        "async_tool_use_stream": {
                            "summary": "Async streaming with tool usage",
                            "value": (
                                "event: message_start\n"
                                'data: {"type":"message_start","message":{"id":"msg_async_67890","type":"message","role":"assistant","content":[],"model":"private-gpt","usage":{}}}\n\n'
                                "event: content_block_start\n"
                                'data: {"type":"content_block_start","block_id":"block_001","content_block":{"type":"text","text":""}}\n\n'
                                "event: content_block_delta\n"
                                'data: {"type":"content_block_delta","block_id":"block_001","delta":{"type":"text_delta","text":"I\'ll help you with that task."}}\n\n'
                                "event: content_block_stop\n"
                                'data: {"type":"content_block_stop","block_id":"block_001"}\n\n'
                                "event: content_block_start\n"
                                'data: {"type":"content_block_start","block_id":"block_002","content_block":{"type":"tool_use","id":"tool_001","name":"security_scanner","input":{}}}\n\n'
                                "event: content_block_delta\n"
                                'data: {"type":"content_block_delta","block_id":"block_002","delta":{"type":"input_json_delta","partial_json":"{\\"directory\\":\\"/src\\"}"}}\n\n'
                                "event: content_block_stop\n"
                                'data: {"type":"content_block_stop","block_id":"block_002"}\n\n'
                                "event: message_delta\n"
                                'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":156,"output_tokens":67}}\n\n'
                                "event: message_stop\n"
                                'data: {"type":"message_stop"}\n\n'
                            ),
                        },
                    }
                }
            },
        },
        404: {
            "description": "Stream not found",
            "content": {
                "application/json": {
                    "examples": {
                        "stream_not_found": {
                            "summary": "Stream does not exist",
                            "value": {
                                "detail": "Stream with message_id msg_nonexistent not found"
                            },
                        }
                    }
                }
            },
        },
    },
)
async def observe_stream(
    request: Request,
    message_id: Annotated[
        str,
        Path(
            description="Message ID of the asynchronous chat stream to observe. ",
            examples=["custom_msg_12345"],
        ),
    ],
) -> StreamingResponse:
    """Observe an asynchronous chat stream via Server-Sent Events.

    This endpoint provides a Server-Sent Events stream that delivers
    chat completion events in real-time. The events follow
    the same format as synchronous chat streaming responses.

    Stream Lifecycle:
    1. Stream initiated via POST /v1/messages/async
    2. Events begin flowing when processing starts
    3. Stream automatically closes when message completes
    4. Stream can be cancelled via POST /v1/messages/async/{message_id}/cancel

    Notes:
    * Stream remains active until message completion or cancellation
    * Events are delivered in chronological order
    * Connection will automatically close when stream ends
    * Use appropriate SSE client libraries for robust event handling
    """
    chat_service: ChatAsyncService = request.state.injector.get(ChatAsyncService)
    event_generator = await chat_service.get_stream_events(message_id)
    if event_generator is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stream with message_id {message_id} not found",
        )

    sse_stream = to_sse_stream(event_generator)
    return StreamingResponse(
        sse_stream,
        media_type="text/event-stream",
    )


@chat_router.get(
    "/{message_id}/status",
    response_model=StreamMetadata,
    summary="Get Async Chat Stream Status",
    responses={
        200: {
            "model": StreamMetadata,
            "description": "Stream status and metadata information",
            "content": {
                "application/json": {
                    "examples": {
                        "pending_stream": {
                            "summary": "Stream in pending state",
                            "value": {
                                "message_id": "msg_async_12345",
                                "status": "pending",
                                "created_at": "2025-07-10T09:11:16.003615Z",
                                "updated_at": "2025-07-10T09:11:16.003615Z",
                                "completed_at": None,
                                "error_message": None,
                                "stream_type": "default",
                                "metadata": {},
                            },
                        },
                        "processing_stream": {
                            "summary": "Stream currently processing",
                            "value": {
                                "message_id": "msg_async_67890",
                                "status": "processing",
                                "created_at": "2025-07-10T09:11:16.003615Z",
                                "updated_at": "2025-07-10T09:11:20.123456Z",
                                "completed_at": None,
                                "error_message": None,
                                "stream_type": "default",
                                "metadata": {
                                    "tokens_processed": 156,
                                    "tools_used": ["security_scanner"],
                                },
                            },
                        },
                        "completed_stream": {
                            "summary": "Successfully completed stream",
                            "value": {
                                "message_id": "msg_async_99999",
                                "status": "completed",
                                "created_at": "2025-07-10T09:11:16.003615Z",
                                "updated_at": "2025-07-10T09:11:28.987654Z",
                                "completed_at": "2025-07-10T09:11:28.987654Z",
                                "error_message": None,
                                "stream_type": "default",
                                "metadata": {
                                    "total_tokens": 521,
                                    "completion_time_ms": 12984,
                                },
                            },
                        },
                        "error_stream": {
                            "summary": "Stream encountered an error",
                            "value": {
                                "message_id": "msg_async_error",
                                "status": "error",
                                "created_at": "2025-07-10T09:11:16.003615Z",
                                "updated_at": "2025-07-10T09:11:18.555555Z",
                                "completed_at": None,
                                "error_message": "Tool execution timeout after 30 seconds",
                                "stream_type": "default",
                                "metadata": {
                                    "error_code": "TOOL_TIMEOUT",
                                    "failed_tool": "security_scanner",
                                },
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Stream not found",
            "content": {
                "application/json": {
                    "examples": {
                        "status_not_found": {
                            "summary": "Stream status not available",
                            "value": {
                                "detail": "Stream with message_id msg_nonexistent not found"
                            },
                        }
                    }
                }
            },
        },
    },
)
async def get_stream_status(
    request: Request,
    message_id: Annotated[
        str,
        Path(
            description=(
                "Message ID of the asynchronous chat stream to check status for. "
            ),
            examples=["custom_msg_12345"],
        ),
    ],
) -> StreamMetadata:
    """Get the current status and metadata of an asynchronous chat stream.

    This endpoint returns comprehensive information about a chat stream's
    current state, including processing status, timestamps, error information,
    and additional metadata collected during processing.

    Status Values:
    * pending: Stream created but processing not yet started
    * processing: Active processing with events being generated
    * completed: Stream finished successfully
    * cancelled: Stream was cancelled by user request
    * error: Stream encountered an error and stopped

    Use Cases:
    * Monitor stream progress without consuming events
    * Check completion status before attempting to observe
    * Debugging failed streams via error messages
    * Performance monitoring through metadata
    """
    chat_service: ChatAsyncService = request.state.injector.get(ChatAsyncService)
    stream_metadata = await chat_service.get_stream_metadata(message_id)
    if stream_metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stream with message_id {message_id} not found",
        )
    return StreamMetadata(
        message_id=message_id,
        **stream_metadata.model_dump(),
    )


@chat_router.post(
    "/{message_id}/cancel",
    summary="Cancel Async Chat Stream",
    response_model=ChatCancellationResponse,
    responses={
        200: {
            "model": ChatCancellationResponse,
            "description": "Stream cancelled successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "cancellation_success": {
                            "summary": "Stream cancelled successfully",
                            "value": {
                                "message": "Stream cancelled successfully",
                                "message_id": "msg_async_12345",
                            },
                        },
                        "cancellation_with_cleanup": {
                            "summary": "Stream cancelled with cleanup",
                            "value": {
                                "message": "Stream cancelled successfully",
                                "message_id": "msg_async_long_running",
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Stream not found",
            "content": {
                "application/json": {
                    "examples": {
                        "cancel_not_found": {
                            "summary": "Cannot cancel non-existent stream",
                            "value": {
                                "detail": "Stream with message_id msg_nonexistent not found"
                            },
                        }
                    }
                }
            },
        },
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    }
                }
            },
            "required": False,
            "description": (
                "Empty request body for stream cancellation. "
                "No additional parameters are required to cancel a stream."
            ),
        }
    },
)
async def cancel_stream(
    request: Request,
    message_id: Annotated[
        str,
        Path(
            description="Message ID of the asynchronous chat stream to cancel. ",
            examples=["custom_msg_12345"],
        ),
    ],
) -> ChatCancellationResponse:
    """Cancel an active asynchronous chat stream.

    This endpoint gracefully cancels an ongoing chat completion stream by:
    1. Setting the cancellation token to stop the event generation loop
    2. Cancelling the underlying asyncio task
    3. Updating the stream status to 'cancelled'

    Notes:
    * Cancellation is irreversible - stream cannot be resumed
    * Any active SSE connections will receive a final event and close
    * Stream status will be updated to reflect cancellation
    * Use DELETE endpoint to remove cancelled streams from storage
    """
    chat_service: ChatAsyncService = request.state.injector.get(ChatAsyncService)
    exist = await chat_service.stream_manager.stream_exists(message_id)
    if not exist:
        raise HTTPException(
            status_code=404,
            detail=f"Stream with message_id {message_id} not found",
        )

    await chat_service.cancel_stream(message_id)
    return ChatCancellationResponse(
        message="Stream cancelled successfully",
        message_id=message_id,
    )


@chat_router.delete(
    "/{message_id}/delete",
    summary="Delete Async Chat Stream",
    responses={
        204: {
            "description": "Stream deleted successfully - no content returned",
        },
        404: {
            "description": "Stream not found",
            "content": {
                "application/json": {
                    "examples": {
                        "delete_not_found": {
                            "summary": "Cannot delete non-existent stream",
                            "value": {
                                "detail": "Stream with message_id msg_nonexistent not found"
                            },
                        }
                    }
                }
            },
        },
    },
)
async def delete_stream(
    request: Request,
    message_id: Annotated[
        str,
        Path(
            description="Message ID of the asynchronous chat stream to delete. ",
            examples=["custom_msg_12345"],
        ),
    ],
) -> None:
    """Delete an asynchronous chat stream and clean up all associated resources.

    This endpoint permanently removes a chat stream from storage and cleans
    up all associated resources including:
    * Stream metadata and status information
    * Cached events and message content
    * Task references and cancellation tokens

    Notes:
    * Stream must exist to be deleted (returns 404 otherwise)
    * Active streams are automatically cancelled before deletion
    * No response body returned on successful deletion (204 status)
    * Use status endpoint to verify deletion if needed
    """
    chat_service: ChatAsyncService = request.state.injector.get(ChatAsyncService)
    exist = await chat_service.stream_manager.stream_exists(message_id)
    if not exist:
        raise HTTPException(
            status_code=404,
            detail=f"Stream with message_id {message_id} not found",
        )

    await chat_service.stream_manager.clean_up_stream(message_id)
