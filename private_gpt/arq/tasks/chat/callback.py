from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)


async def resume_chat_callback(
    *,
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    correlation_id = request.context.get("correlation_id")
    if not correlation_id or not request.tool_id:
        return

    from private_gpt.arq.enqueue import enqueue_tool_resume_job

    await enqueue_tool_resume_job(
        correlation_id=correlation_id,
        tool_id=request.tool_id,
        result=response.model_dump(mode="json"),
    )
