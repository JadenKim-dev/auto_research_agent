import pytest
import os
from unittest.mock import Mock, patch
from typing import List
import fakeredis
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'REDIS_URL': 'redis://localhost:6379/0'
    }):
        yield


@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI instance."""
    llm = Mock(spec=ChatOpenAI)
    llm.model_name = "gpt-4o-mini"
    llm.invoke.return_value = Mock(content="Test summary response")
    return llm


@pytest.fixture
def fake_redis():
    """Fake Redis instance for testing."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def test_session_id():
    """Test session ID."""
    return "test_session_123"


@pytest.fixture
def test_messages() -> List[BaseMessage]:
    """Sample messages for testing."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        HumanMessage(content="What's the weather like?"),
        AIMessage(content="I don't have access to current weather data."),
        SystemMessage(content="Summary: User asked about weather"),
    ]


@pytest.fixture
def large_message_list() -> List[BaseMessage]:
    """Large message list for testing summarization."""
    messages = []
    for i in range(25):  # More than MAX_MESSAGES (20)
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"User message {i}"))
        else:
            messages.append(AIMessage(content=f"AI response {i}"))
    return messages


@pytest.fixture
def mock_summary_manager():
    """Mock ConversationSummaryManager."""
    manager = Mock()
    manager.summarize_messages.return_value = "Test summary of conversation"
    return manager


@pytest.fixture
def mock_redis_client():
    """Mock Redis client that behaves like FakeRedis but can be easily controlled."""
    client = Mock()
    client.get.return_value = None
    client.setex.return_value = True
    client.delete.return_value = 1
    client.pipeline.return_value = client
    client.execute.return_value = [True]
    client.ping.return_value = True
    return client


# Clean up singletons after each test
@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Clean up singleton instances after each test."""
    yield
    # Reset singleton instances
    from app.core.llm_client import LLMClient
    from app.core.redis_client import RedisClient
    
    LLMClient._instance = None
    RedisClient._instance = None
    RedisClient._pool = None