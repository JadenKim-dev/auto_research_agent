import asyncio
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from ..schemas.agent import StreamingEvent


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
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
        event = StreamingEvent(
            type="final_answer",
            data={"output": finish.return_values.get("output", "")},
        )
        asyncio.run_coroutine_threadsafe(self.queue.put(event.to_json()), self.loop)
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)
