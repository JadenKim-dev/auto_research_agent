from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
import logging

from .prompts import react_prompt, research_react_prompt, simple_react_prompt
from ..tools.basic_tools import BASIC_TOOLS
from ..memory.redis_memory import RedisConversationSummaryMemory
from ..core.redis_client import RedisClient
from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Available prompt types for the ReAct agent."""

    STANDARD = "standard"
    RESEARCH = "research"
    SIMPLE = "simple"


class ReActCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[str] = []

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.actions.append(
            {"tool": action.tool, "input": action.tool_input, "log": action.log}
        )
        logger.info(f"Agent Action: {action.tool} with input: {action.tool_input}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        logger.info(f"Agent Finish: {finish.return_values}")

    def get_trace(self) -> Dict[str, List]:
        return {
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
        }


class ResearchReActAgent:
    def __init__(
        self,
        session_id: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[ChatOpenAI] = None,
        prompt_type: PromptType = PromptType.STANDARD,
        memory: Optional[RedisConversationSummaryMemory] = None,
        verbose: bool = True,
        max_iterations: int = 10,
        early_stopping_method: Literal["force", "generate"] = "force",
        **kwargs,
    ):
        """
        Args:
            tools: List of tools available to the agent
            llm: Language model to use (defaults to OpenAI)
            prompt_type: Type of prompt to use (PromptType enum)
            memory: Redis conversation memory (will be created if not provided)
            session_id: Session ID for Redis memory (required)
            verbose: Whether to print reasoning steps
            max_iterations: Maximum number of reasoning steps
            early_stopping_method: Method to use when stopping early ('force' or 'generate')
        """
        self.llm = llm or LLMClient.get_client()

        self.tools = tools or BASIC_TOOLS

        prompt_map = {
            PromptType.STANDARD: react_prompt,
            PromptType.RESEARCH: research_react_prompt,
            PromptType.SIMPLE: simple_react_prompt,
        }
        self.prompt = prompt_map.get(prompt_type, react_prompt)

        if memory:
            self.memory = memory
        else:
            if not session_id:
                raise ValueError("session_id is required")

            redis_client = RedisClient.get_client()
            self.memory = RedisConversationSummaryMemory(
                session_id=session_id,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
                redis_client=redis_client,
                llm=self.llm,
            )
            logger.info(f"Using Redis-backed summary memory for session: {session_id}")

        self.agent = create_react_agent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,
            max_iterations=max_iterations,
            early_stopping_method=early_stopping_method,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        self.config = {
            "prompt_type": prompt_type.value,
            "verbose": verbose,
            "max_iterations": max_iterations,
            "early_stopping_method": early_stopping_method,
            "model": self.llm.model_name,
        }

    async def run(
        self, query: str, callbacks: Optional[List[BaseCallbackHandler]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent with a query.

        Args:
            query: The input question or task
            callbacks: Optional list of callback handlers

        Returns:
            Dictionary containing the result and metadata
        """
        if callbacks is None:
            callbacks = [ReActCallbackHandler()]
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
        for callback in callbacks:
            if isinstance(callback, ReActCallbackHandler):
                trace = callback.get_trace()
                break

        return {
            "output": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", []),
            "trace": trace,
            "config": self.config,
        }


def create_research_agent(
    session_id: str,
    prompt_type: PromptType = PromptType.RESEARCH,
    tools: Optional[List[BaseTool]] = None,
    **kwargs,
) -> ResearchReActAgent:
    """
    Factory function to create a research agent.

    Args:
        prompt_type: Type of prompt to use (PromptType enum)
        tools: Optional list of tools
        session_id: Session ID for Redis memory (required)
        **kwargs: Additional arguments for the agent

    Returns:
        Configured ResearchReActAgent instance
    """
    return ResearchReActAgent(
        tools=tools,
        prompt_type=prompt_type,
        session_id=session_id,
        **kwargs,
    )


__all__ = [
    "ResearchReActAgent",
    "create_research_agent",
    "ReActCallbackHandler",
    "PromptType",
]
