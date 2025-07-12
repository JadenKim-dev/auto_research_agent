from fastapi import APIRouter
import asyncio
from sse_starlette.sse import EventSourceResponse
import uuid

from ...agents.react_agent import create_research_agent
from ..schemas.agent import AgentQueryRequest, StreamingEvent
from ..handlers.streaming import StreamingCallbackHandler

router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)


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

        session_event = StreamingEvent(type="session", data={"session_id": session_id})
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

        result = await task

        final_event = StreamingEvent(
            type="complete",
            data={"result": result},
        )
        yield f"data: {final_event.to_json()}\n\n"

    return EventSourceResponse(event_generator())


__all__ = ["router"]
