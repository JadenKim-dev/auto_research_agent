from typing import List, Dict, Any, Optional
from enum import Enum
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field, field_validator
import logging

from .prompts import react_prompt, research_react_prompt, simple_react_prompt
from ..tools.basic_tools import BASIC_TOOLS
from ..memory.redis_memory import RedisChatMessageHistory
from ..memory.summary_manager import ConversationSummaryManager
from ..core.redis_client import RedisClient
from ..core.llm_client import LLMClient
from ..logging.tao_logger import TAOLogger

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Available prompt types for the ReAct agent."""

    STANDARD = "standard"
    RESEARCH = "research"
    SIMPLE = "simple"


# Prompt mapping - defined at module level for efficiency
PROMPT_MAP = {
    PromptType.STANDARD: react_prompt,
    PromptType.RESEARCH: research_react_prompt,
    PromptType.SIMPLE: simple_react_prompt,
}


class AgentConfig(BaseModel):    
    session_id: str = Field(..., description="Session ID for Redis memory")
    prompt_type: PromptType = Field(
        default=PromptType.STANDARD,
        description="Type of prompt to use"
    )

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if not v or not v.strip():
            raise ValueError("session_id is required and cannot be empty")
        return v.strip()

    @field_validator('prompt_type')
    @classmethod
    def validate_prompt_type(cls, v):
        if not isinstance(v, PromptType):
            raise TypeError(f"prompt_type must be a PromptType enum, got {type(v)}")
        return v


class ResearchReActAgent:
    def __init__(
        self,
        session_id: str,
        prompt_type: PromptType = PromptType.STANDARD,
    ):
        """
        Args:
            session_id: Session ID for Redis memory (required)
            prompt_type: Type of prompt to use (PromptType enum)
        """
        config = AgentConfig(session_id=session_id, prompt_type=prompt_type)
        
        self.session_id = config.session_id
        self.prompt_type = config.prompt_type
        self.llm = LLMClient.get_client()
        self.tools = BASIC_TOOLS

        self.prompt = PROMPT_MAP.get(config.prompt_type, react_prompt)

        redis_client = RedisClient.get_client()
        self.summary_manager = ConversationSummaryManager(llm=self.llm)
        self.memory = RedisChatMessageHistory(
            session_id=self.session_id,
            redis_client=redis_client,
            summary_manager=self.summary_manager,
        )
        logger.info(f"Using Redis-backed chat history for session: {self.session_id}")

        self.agent = create_react_agent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            early_stopping_method="force",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        self.config = {
            "prompt_type": config.prompt_type.value,
            "verbose": True,
            "max_iterations": 10,
            "early_stopping_method": "force",
            "model": self.llm.model_name,
        }

    async def run(
        self,
        query: str,
        callbacks: List[BaseCallbackHandler] = [],
    ) -> Dict[str, Any]:
        """
        Run the agent with a query.

        Args:
            query: The input question or task
            callbacks: Optional list of callback handlers

        Returns:
            Dictionary containing the result and metadata
        """
        tao_logger = TAOLogger(
            session_id=getattr(self.memory, "session_id", "default")
        )
        callbacks.append(tao_logger)

        try:
            result = await self.executor.ainvoke(
                {"input": query}, config={"callbacks": callbacks}
            )
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return {
                "output": f"Error: {str(e)}",
                "intermediate_steps": [],
                "trace": None,
                "config": self.config,
                "error": True,
            }

        trace = None
        tao_logs = None
        for callback in callbacks:
            if isinstance(callback, TAOLogger):
                trace = callback.get_trace()
                tao_logs = callback.get_logs()

        return {
            "output": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", []),
            "trace": trace,
            "tao_logs": tao_logs,
            "config": self.config,
        }


__all__ = [
    "ResearchReActAgent",
    "PromptType",
]
