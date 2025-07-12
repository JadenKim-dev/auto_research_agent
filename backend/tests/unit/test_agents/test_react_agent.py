import pytest
from unittest.mock import Mock, patch
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from app.agents.react_agent import (
    ResearchReActAgent,
    create_research_agent,
    PromptType,
    ReActCallbackHandler,
)
from app.memory.redis_memory import RedisChatMessageHistory
from typing import List, cast


class TestReActCallbackHandler:
    """Test cases for ReActCallbackHandler."""

    def test_init(self):
        """Test ReActCallbackHandler initialization."""
        handler = ReActCallbackHandler()
        assert handler.thoughts == []

    def test_on_agent_action(self):
        """Test on_agent_action method stores thoughts."""
        handler = ReActCallbackHandler()
        mock_action = Mock()
        mock_action.log = "I need to search for information"

        handler.on_agent_action(mock_action, color="blue")

        assert len(handler.thoughts) == 1
        assert handler.thoughts[0] == "I need to search for information"

    def test_on_agent_finish(self):
        """Test on_agent_finish method stores final thought."""
        handler = ReActCallbackHandler()
        mock_finish = Mock()
        mock_finish.log = "I have found the answer"

        handler.on_agent_finish(mock_finish, color="green")

        assert len(handler.thoughts) == 1
        assert handler.thoughts[0] == "I have found the answer"

    def test_on_tool_end(self):
        """Test on_tool_end method stores observations."""
        handler = ReActCallbackHandler()
        tool_output = "Search results: 42 is the answer"

        handler.on_tool_end(tool_output)

        assert len(handler.observations) == 1
        assert handler.observations[0] == "Search results: 42 is the answer"


class TestResearchReActAgent:
    """Test cases for ResearchReActAgent."""

    @patch("app.agents.react_agent.create_react_agent")
    @patch("app.agents.react_agent.AgentExecutor")
    def test_init_with_memory(
        self, mock_executor_class, mock_create_agent, mock_llm, test_session_id
    ):
        """Test agent initialization with provided memory."""
        mock_memory = Mock(spec=RedisChatMessageHistory)
        mock_tools = cast(List[BaseTool], [Mock(spec=BaseTool)])

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor

        agent = ResearchReActAgent(
            session_id=test_session_id,
            tools=mock_tools,
            llm=mock_llm,
            memory=mock_memory,
            verbose=True,
            max_iterations=15,
        )

        assert agent.llm == mock_llm
        assert agent.tools == mock_tools
        assert agent.memory == mock_memory
        assert agent.agent == mock_agent
        assert agent.executor == mock_executor

        # Verify AgentExecutor was called with correct parameters
        mock_executor_class.assert_called_once()
        call_kwargs = mock_executor_class.call_args[1]
        assert call_kwargs["verbose"] == True
        assert call_kwargs["max_iterations"] == 15

    @patch("app.agents.react_agent.create_react_agent")
    @patch("app.agents.react_agent.AgentExecutor")
    @patch("app.agents.react_agent.RedisClient")
    @patch("app.agents.react_agent.LLMClient")
    def test_init_without_memory(
        self,
        mock_llm_client,
        mock_redis_client,
        mock_executor_class,
        mock_create_agent,
        test_session_id,
    ):
        """Test agent initialization without provided memory (creates new one)."""
        # Setup mocks
        mock_llm = Mock(spec=ChatOpenAI)
        mock_llm.model_name = "gpt-4o-mini"
        mock_llm_client.get_client.return_value = mock_llm

        mock_redis = Mock()
        mock_redis_client.get_client.return_value = mock_redis

        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor

        with patch(
            "app.agents.react_agent.RedisChatMessageHistory"
        ) as mock_memory_class:
            with patch(
                "app.agents.react_agent.ConversationSummaryManager"
            ) as mock_summary_class:
                mock_memory = Mock()
                mock_memory_class.return_value = mock_memory
                mock_summary_manager = Mock()
                mock_summary_class.return_value = mock_summary_manager

                agent = ResearchReActAgent(
                    session_id=test_session_id, prompt_type=PromptType.RESEARCH
                )

                # Verify memory and summary manager were created
                mock_summary_class.assert_called_once_with(llm=mock_llm)
                mock_memory_class.assert_called_once_with(
                    session_id=test_session_id,
                    redis_client=mock_redis,
                    summary_manager=mock_summary_manager,
                )

                assert agent.memory == mock_memory
                assert agent.summary_manager == mock_summary_manager

    def test_init_missing_session_id(self):
        """Test that missing session_id raises ValueError."""
        with pytest.raises(ValueError, match="session_id is required"):
            ResearchReActAgent(session_id=None)  # type: ignore

    @patch("app.agents.react_agent.create_react_agent")
    @patch("app.agents.react_agent.AgentExecutor")
    def test_prompt_type_selection(
        self, mock_executor_class, mock_create_agent, mock_llm, test_session_id
    ):
        """Test that different prompt types are selected correctly."""
        mock_memory = Mock(spec=RedisChatMessageHistory)

        with patch(
            "app.agents.react_agent.research_react_prompt"
        ) as mock_research_prompt:
            agent = ResearchReActAgent(
                session_id=test_session_id,
                llm=mock_llm,
                memory=mock_memory,
                prompt_type=PromptType.RESEARCH,
            )

            # Verify the research prompt was used
            mock_create_agent.assert_called_once()
            call_args = mock_create_agent.call_args[1]
            assert call_args["prompt"] == mock_research_prompt

    @pytest.mark.asyncio
    async def test_run_success(self, mock_llm, test_session_id):
        """Test successful agent run."""
        mock_memory = Mock(spec=RedisChatMessageHistory)

        with patch("app.agents.react_agent.create_react_agent"):
            with patch("app.agents.react_agent.AgentExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor

                # Mock successful execution
                mock_result = {
                    "output": "The answer is 42",
                    "intermediate_steps": [("thought", "action")],
                }

                async def mock_ainvoke(*args, **kwargs):
                    return mock_result

                mock_executor.ainvoke = mock_ainvoke

                agent = ResearchReActAgent(
                    session_id=test_session_id, llm=mock_llm, memory=mock_memory
                )

                result = await agent.run("What is the answer?")

                assert result["output"] == "The answer is 42"
                assert result["intermediate_steps"] == [("thought", "action")]
                assert "config" in result

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self, mock_llm, test_session_id):
        """Test agent run with custom callbacks."""
        mock_memory = Mock(spec=RedisChatMessageHistory)
        custom_callback = Mock()

        with patch("app.agents.react_agent.create_react_agent"):
            with patch("app.agents.react_agent.AgentExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                mock_executor.ainvoke.return_value = {"output": "result"}

                agent = ResearchReActAgent(
                    session_id=test_session_id, llm=mock_llm, memory=mock_memory
                )

                await agent.run("test query", callbacks=[custom_callback])

                # Verify custom callback was used
                call_args = mock_executor.ainvoke.call_args[1]
                assert custom_callback in call_args["config"]["callbacks"]

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, mock_llm, test_session_id):
        """Test that exceptions during execution are handled properly."""
        mock_memory = Mock(spec=RedisChatMessageHistory)

        with patch("app.agents.react_agent.create_react_agent"):
            with patch("app.agents.react_agent.AgentExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor

                # Mock execution failure
                mock_executor.ainvoke.side_effect = Exception("Execution failed")

                agent = ResearchReActAgent(
                    session_id=test_session_id, llm=mock_llm, memory=mock_memory
                )

                result = await agent.run("test query")

                assert result["error"] == True
                assert "Error: Execution failed" in result["output"]
                assert result["intermediate_steps"] == []

    def test_config_property(self, mock_llm, test_session_id):
        """Test that config property contains correct information."""
        mock_memory = Mock(spec=RedisChatMessageHistory)

        with patch("app.agents.react_agent.create_react_agent"):
            with patch("app.agents.react_agent.AgentExecutor"):
                agent = ResearchReActAgent(
                    session_id=test_session_id,
                    llm=mock_llm,
                    memory=mock_memory,
                    prompt_type=PromptType.SIMPLE,
                    verbose=False,
                    max_iterations=20,
                    early_stopping_method="generate",
                )

                config = agent.config

                assert config["prompt_type"] == "simple"
                assert config["verbose"] == False
                assert config["max_iterations"] == 20
                assert config["early_stopping_method"] == "generate"
                assert config["model"] == mock_llm.model_name


class TestCreateResearchAgent:
    """Test cases for create_research_agent function."""

    @patch("app.agents.react_agent.ResearchReActAgent")
    def test_create_research_agent(self, mock_agent_class, test_session_id):
        """Test create_research_agent function."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        result = create_research_agent(
            session_id=test_session_id,
            prompt_type=PromptType.RESEARCH,
            max_iterations=25,
        )

        assert result == mock_agent
        mock_agent_class.assert_called_once_with(
            tools=None,
            prompt_type=PromptType.RESEARCH,
            session_id=test_session_id,
            max_iterations=25,
        )
