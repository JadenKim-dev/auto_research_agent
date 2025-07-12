from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import asyncio
from sse_starlette.sse import EventSourceResponse
from langchain.schema import AgentAction, AgentFinish

from ...agents.react_agent import create_research_agent, ReActCallbackHandler, PromptType

router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="The question or task for the agent")
    prompt_type: PromptType = Field(
        default=PromptType.STANDARD,
        description=f"Type of prompt to use: {', '.join(PromptType)}",
    )
    max_iterations: int = Field(
        default=10, description="Maximum number of reasoning steps"
    )
    early_stopping_method: Literal["force", "generate"] = Field(
        default="force", 
        description="Method to use when stopping early: 'force' (stop immediately) or 'generate' (generate final answer)"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


class AgentQueryResponse(BaseModel):
    output: str = Field(..., description="The agent's final answer")
    intermediate_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="The agent's reasoning steps"
    )
    trace: Optional[Dict[str, List]] = Field(
        default=None, description="Detailed trace of thoughts and actions"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Agent configuration used"
    )
    error: bool = Field(default=False, description="Whether an error occurred")


class ToolInfo(BaseModel):
    name: str
    description: str


class AgentInfoResponse(BaseModel):
    available_tools: List[ToolInfo]
    prompt_types: List[str]
    model: str


# Streaming Event DTO
class StreamingEvent(BaseModel):
    type: Literal["action", "observation", "final_answer", "error", "complete"]
    data: Dict[str, Any] = Field(..., description="Event data")

    def to_json(self) -> str:
        return self.model_dump_json()


# Global agent instance
_agent = None


def get_agent(early_stopping_method: Literal["force", "generate"] = "force"):
    global _agent
    if _agent is None:
        _agent = create_research_agent(
            prompt_type=PromptType.STANDARD,
            early_stopping_method=early_stopping_method
        )
    return _agent


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
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)  # Signal end of stream


@router.post("/query/stream")
async def query_agent_stream(request: AgentQueryRequest):
    """
    Submit a query to the ReAct agent with streaming response.

    Returns a stream of Server-Sent Events with the agent's reasoning process.
    """

    async def event_generator():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        agent = create_research_agent(
            prompt_type=request.prompt_type,
            max_iterations=request.max_iterations,
            early_stopping_method=request.early_stopping_method,
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


@router.get("/info", response_model=AgentInfoResponse)
async def get_agent_info():
    """
    Get information about the agent and available tools.
    """
    agent = get_agent()

    return AgentInfoResponse(
        available_tools=[
            ToolInfo(name=tool.name, description=tool.description)
            for tool in agent.tools
        ],
        prompt_types=[prompt_type.value for prompt_type in PromptType],
        model=agent.llm.model_name,
    )


@router.post("/clear-memory")
async def clear_agent_memory():
    """
    Clear the agent's conversation memory.
    """
    agent = get_agent()
    agent.clear_memory()

    return {"message": "Agent memory cleared successfully"}


__all__ = ["router"]
