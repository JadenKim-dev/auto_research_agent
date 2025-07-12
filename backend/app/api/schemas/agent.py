from pydantic import BaseModel, Field
from typing import Dict, Any, Literal

from ...agents.react_agent import PromptType


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="The question or task for the agent")
    session_id: str = Field(
        default="", description="Session ID for conversation continuity"
    )
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


class StreamingEvent(BaseModel):
    type: Literal[
        "session", "action", "observation", "final_answer", "error", "complete"
    ]
    data: Dict[str, Any] = Field(..., description="Event data")

    def to_json(self) -> str:
        return self.model_dump_json()
