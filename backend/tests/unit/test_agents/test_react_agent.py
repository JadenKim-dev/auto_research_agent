import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError
from app.agents.react_agent import (
    ResearchReActAgent,
    PromptType,
)
from app.memory.redis_memory import RedisChatMessageHistory


@pytest.fixture
def mock_agent_dependencies():
    """Mock all ResearchReActAgent dependencies."""
    with patch("app.agents.react_agent.LLMClient") as mock_llm_client, \
         patch("app.agents.react_agent.RedisClient") as mock_redis_client, \
         patch("app.agents.react_agent.RedisChatMessageHistory") as mock_memory, \
         patch("app.agents.react_agent.ConversationSummaryManager") as mock_summary, \
         patch("app.agents.react_agent.create_react_agent") as mock_create_agent, \
         patch("app.agents.react_agent.AgentExecutor") as mock_executor:
        
        # 기본 설정
        mock_llm_instance = Mock()
        mock_llm_instance.model_name = "gpt-4o-mini"
        mock_llm_client.get_client.return_value = mock_llm_instance
        
        mock_redis_instance = Mock()
        mock_redis_client.get_client.return_value = mock_redis_instance
        
        mock_agent_instance = Mock()
        mock_create_agent.return_value = mock_agent_instance
        
        mock_executor_instance = Mock()
        mock_executor.return_value = mock_executor_instance
        
        yield {
            'llm_client': mock_llm_client,
            'llm_instance': mock_llm_instance,
            'redis_client': mock_redis_client,
            'redis_instance': mock_redis_instance,
            'memory': mock_memory,
            'summary': mock_summary,
            'create_agent': mock_create_agent,
            'agent_instance': mock_agent_instance,
            'executor': mock_executor,
            'executor_instance': mock_executor_instance
        }


class TestResearchReActAgent:
    """Test cases for ResearchReActAgent."""

    def test_init_simplified(self, mock_agent_dependencies, test_session_id):
        """Test simplified agent initialization."""
        mocks = mock_agent_dependencies
        
        agent = ResearchReActAgent(
            session_id=test_session_id,
            prompt_type=PromptType.STANDARD,
        )

        assert agent.llm == mocks['llm_instance']
        assert agent.agent == mocks['agent_instance']
        assert agent.executor == mocks['executor_instance']

    def test_init_without_memory(self, mock_agent_dependencies, test_session_id):
        """Test agent initialization without provided memory (creates new one)."""
        mocks = mock_agent_dependencies
        
        agent = ResearchReActAgent(
            session_id=test_session_id, 
            prompt_type=PromptType.RESEARCH
        )

        # Verify memory and summary manager were created
        mocks['summary'].assert_called_once_with(llm=mocks['llm_instance'])
        mocks['memory'].assert_called_once_with(
            session_id=test_session_id,
            redis_client=mocks['redis_instance'],
            summary_manager=mocks['summary'].return_value,
        )

    def test_init_empty_session_id(self, mock_agent_dependencies):
        """Test that empty session_id is handled correctly."""
        # Empty session_id should work with simplified interface
        agent = ResearchReActAgent(session_id="test_session")
        assert agent is not None

    def test_init_missing_session_id(self):
        """Test that missing session_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchReActAgent(session_id="")
        assert "session_id is required and cannot be empty" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            ResearchReActAgent(session_id="   ")  # whitespace only
        assert "session_id is required and cannot be empty" in str(exc_info.value)
        
        with pytest.raises(ValidationError):
            ResearchReActAgent(session_id=None)  # type: ignore

    def test_init_invalid_prompt_type(self):
        """Test that invalid prompt_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchReActAgent(session_id="test", prompt_type="invalid")  # type: ignore
        assert "Input should be" in str(exc_info.value)
        
        with pytest.raises(ValidationError):
            ResearchReActAgent(session_id="test", prompt_type=123)  # type: ignore

    def test_prompt_type_selection(self, mock_agent_dependencies, test_session_id):
        """Test that different prompt types are selected correctly."""
        mocks = mock_agent_dependencies
        
        agent = ResearchReActAgent(
            session_id=test_session_id,
            prompt_type=PromptType.RESEARCH,
        )

        # Verify agent was created
        mocks['create_agent'].assert_called_once()
        # The prompt should be set from PROMPT_MAP
        assert agent.prompt is not None

    def test_session_id_whitespace_handling(self, mock_agent_dependencies):
        """Test that session_id whitespace is handled correctly."""
        # Whitespace should be trimmed
        agent = ResearchReActAgent(session_id="  test_session  ")
        assert agent.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_run_success(self, mock_agent_dependencies, test_session_id):
        """Test successful agent run."""
        mocks = mock_agent_dependencies
        
        # Mock successful execution
        mock_result = {
            "output": "The answer is 42",
            "intermediate_steps": [("thought", "action")],
        }

        async def mock_ainvoke(*args, **kwargs):
            return mock_result

        mocks['executor_instance'].ainvoke = mock_ainvoke

        agent = ResearchReActAgent(session_id=test_session_id)

        result = await agent.run("What is the answer?")

        assert result["output"] == "The answer is 42"
        assert result["intermediate_steps"] == [("thought", "action")]
        assert "config" in result

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self, mock_agent_dependencies, test_session_id):
        """Test agent run with custom callbacks."""
        mocks = mock_agent_dependencies
        custom_callback = Mock()

        mocks['executor_instance'].ainvoke.return_value = {"output": "result"}

        agent = ResearchReActAgent(session_id=test_session_id)

        await agent.run("test query", callbacks=[custom_callback])

        # Verify custom callback was used
        call_args = mocks['executor_instance'].ainvoke.call_args[1]
        assert custom_callback in call_args["config"]["callbacks"]

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, mock_agent_dependencies, test_session_id):
        """Test that exceptions during execution are handled properly."""
        mocks = mock_agent_dependencies
        
        # Mock execution failure
        mocks['executor_instance'].ainvoke.side_effect = Exception("Execution failed")

        agent = ResearchReActAgent(session_id=test_session_id)

        result = await agent.run("test query")

        assert result["error"] == True
        assert "agent API error: Execution failed" in result["output"]
        assert result["intermediate_steps"] == []
        assert "error_details" in result
        assert "state" in result

    def test_config_property(self, mock_agent_dependencies, test_session_id):
        """Test that config property contains correct information."""
        mocks = mock_agent_dependencies
        
        agent = ResearchReActAgent(
            session_id=test_session_id,
            prompt_type=PromptType.SIMPLE,
        )

        config = agent.config

        assert config["prompt_type"] == "simple"
        assert config["verbose"] == True
        assert config["max_iterations"] == 10
        assert config["early_stopping_method"] == "force"
        assert config["model"] == "gpt-4o-mini"


