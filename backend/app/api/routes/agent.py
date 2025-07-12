from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal
import asyncio
from sse_starlette.sse import EventSourceResponse
from langchain.schema import AgentAction, AgentFinish
import uuid

from ...agents.react_agent import (
    ResearchReActAgent,
    create_research_agent,
    ReActCallbackHandler,
    PromptType,
)

router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="The question or task for the agent")
    session_id: str = Field(default="", description="Session ID for conversation continuity")
    prompt_type: PromptType = Field(
        default=PromptType.STANDARD,
        description=f"Type of prompt to use: {', '.join(PromptType)}",
    )
    max_iterations: int = Field(
        default=10, gt=0, description="Maximum number of reasoning steps"
    )
    early_stopping_method: Literal["force", "generate"] = Field(
        default="force", description="Method to use when stopping early"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


# Streaming Event DTO
class StreamingEvent(BaseModel):
    type: Literal["session", "action", "observation", "final_answer", "error", "complete"]
    data: Dict[str, Any] = Field(..., description="Event data")

    def to_json(self) -> str:
        return self.model_dump_json()


class StreamingCallbackHandler(ReActCallbackHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        super().on_agent_action(action, **kwargs)
        event = StreamingEvent(
            type="action",
            data={
                "tool": action.tool,
                "input": action.tool_input,
                "thought": action.log,
            },
        )
        asyncio.run_coroutine_threadsafe(self.queue.put(event.to_json()), self.loop)

    def on_tool_end(self, output: str, **kwargs) -> None:
        event = StreamingEvent(
            type="observation",
            data={"output": output},
        )
        asyncio.run_coroutine_threadsafe(self.queue.put(event.to_json()), self.loop)

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        super().on_agent_finish(finish, **kwargs)
        event = StreamingEvent(
            type="final_answer",
            data={"output": finish.return_values.get("output", "")},
        )
        asyncio.run_coroutine_threadsafe(self.queue.put(event.to_json()), self.loop)
        asyncio.run_coroutine_threadsafe(
            self.queue.put(None), self.loop
        )  # Signal end of stream


@router.post("/query")
async def query_agent_stream(request: AgentQueryRequest):
    """
    Submit a query to the ReAct agent with streaming response.

    Returns a stream of Server-Sent Events with the agent's reasoning process.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        session_event = StreamingEvent(
            type="session",
            data={"session_id": session_id}
        )
        yield f"data: {session_event.to_json()}\n\n"

        agent = create_research_agent(
            prompt_type=request.prompt_type,
            max_iterations=request.max_iterations,
            early_stopping_method=request.early_stopping_method,
            session_id=session_id,
            verbose=False,  # Disable verbose to avoid duplicate logs
        )

        callback = StreamingCallbackHandler(queue, loop)

        task = asyncio.create_task(agent.run(request.query, callbacks=[callback]))
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {event}\n\n"

        try:
            result = await task
        except Exception as e:
            error_event = StreamingEvent(
                type="error",
                data={"error": str(e)},
            )
            yield f"data: {error_event.to_json()}\n\n"
            return

        final_event = StreamingEvent(
            type="complete",
            data={"result": result},
        )
        yield f"data: {final_event.to_json()}\n\n"

    return EventSourceResponse(event_generator())


__all__ = ["router"]
