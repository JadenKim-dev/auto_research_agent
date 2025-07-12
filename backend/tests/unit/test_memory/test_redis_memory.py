import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.memory.redis_memory import RedisChatMessageHistory


class TestRedisChatMessageHistory:
    """Test cases for RedisChatMessageHistory."""

    def test_init(self, fake_redis, mock_summary_manager, test_session_id):
        """Test RedisChatMessageHistory initialization."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager,
            ttl=3600
        )
        
        assert memory.session_id == test_session_id
        assert memory.redis_client == fake_redis
        assert memory.summary_manager == mock_summary_manager
        assert memory.ttl == 3600
        assert memory._messages == []

    def test_init_default_ttl(self, fake_redis, mock_summary_manager, test_session_id):
        """Test initialization with default TTL."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        assert memory.ttl == 86400  # 24 hours

    def test_get_redis_key(self, fake_redis, mock_summary_manager, test_session_id):
        """Test Redis key generation."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        expected_key = f"chat_history:{test_session_id}"
        assert memory._get_redis_key() == expected_key

    def test_messages_property(self, fake_redis, mock_summary_manager, test_session_id, test_messages):
        """Test messages property returns internal messages list."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        memory._messages = test_messages
        
        assert memory.messages == test_messages

    def test_add_message_no_summarization(self, fake_redis, mock_summary_manager, test_session_id):
        """Test adding message without triggering summarization."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        message = HumanMessage(content="Hello world")
        memory.add_message(message)
        
        assert len(memory._messages) == 1
        assert memory._messages[0] == message
        mock_summary_manager.summarize_messages.assert_not_called()

    def test_add_message_triggers_summarization(self, fake_redis, mock_summary_manager, test_session_id):
        """Test adding message triggers summarization when limit exceeded."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        # Add MAX_MESSAGES (20) messages first
        for i in range(memory.MAX_MESSAGES):
            memory._messages.append(HumanMessage(content=f"Message {i}"))
        
        mock_summary_manager.summarize_messages.return_value = "Summary of first 10 messages"
        
        # Add one more message to trigger summarization
        new_message = HumanMessage(content="Trigger message")
        memory.add_message(new_message)
        
        # Should have: 1 SystemMessage + (20 - 10) original + 1 new = 12 messages
        assert len(memory._messages) == 12
        assert isinstance(memory._messages[0], SystemMessage)
        assert "Summary: Summary of first 10 messages" in memory._messages[0].content
        
        # Verify summarization was called with first 10 messages
        mock_summary_manager.summarize_messages.assert_called_once()
        summarized_messages = mock_summary_manager.summarize_messages.call_args[0][0]
        assert len(summarized_messages) == memory.SUMMARY_BATCH_SIZE

    def test_check_and_summarize_no_action_under_limit(self, fake_redis, mock_summary_manager, test_session_id):
        """Test _check_and_summarize does nothing when under message limit."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        # Add fewer than MAX_MESSAGES
        for i in range(5):
            memory._messages.append(HumanMessage(content=f"Message {i}"))
        
        memory._check_and_summarize()
        
        assert len(memory._messages) == 5
        mock_summary_manager.summarize_messages.assert_not_called()

    def test_serialize_message(self, fake_redis, mock_summary_manager, test_session_id):
        """Test message serialization."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        human_msg = HumanMessage(content="Hello", additional_kwargs={"user_id": "123"})
        ai_msg = AIMessage(content="Hi there")
        system_msg = SystemMessage(content="Summary: Previous conversation")
        
        human_serialized = memory._serialize_message(human_msg)
        ai_serialized = memory._serialize_message(ai_msg)
        system_serialized = memory._serialize_message(system_msg)
        
        assert human_serialized == {
            "type": "HumanMessage",
            "content": "Hello",
            "additional_kwargs": {"user_id": "123"}
        }
        
        assert ai_serialized == {
            "type": "AIMessage",
            "content": "Hi there",
            "additional_kwargs": {}
        }
        
        assert system_serialized == {
            "type": "SystemMessage",
            "content": "Summary: Previous conversation",
            "additional_kwargs": {}
        }

    def test_save_to_redis(self, fake_redis, mock_summary_manager, test_session_id, test_messages):
        """Test saving messages to Redis."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        memory._messages = test_messages
        
        memory._save_to_redis()
        
        # Check that data was saved to Redis
        saved_data = fake_redis.get(memory._get_redis_key())
        assert saved_data is not None
        
        data = json.loads(saved_data)
        assert "messages" in data
        assert "timestamp" in data
        assert len(data["messages"]) == len(test_messages)

    def test_load_from_redis_empty(self, fake_redis, mock_summary_manager, test_session_id):
        """Test loading from Redis when no data exists."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        # Should have no messages since Redis is empty
        assert memory._messages == []

    def test_load_from_redis_with_data(self, fake_redis, mock_summary_manager, test_session_id):
        """Test loading existing data from Redis."""
        # Pre-populate Redis with test data
        test_data = {
            "messages": [
                {"type": "HumanMessage", "content": "Hello", "additional_kwargs": {}},
                {"type": "AIMessage", "content": "Hi", "additional_kwargs": {}},
                {"type": "SystemMessage", "content": "Summary", "additional_kwargs": {}}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        fake_redis.set(f"chat_history:{test_session_id}", json.dumps(test_data))
        
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        assert len(memory._messages) == 3
        assert isinstance(memory._messages[0], HumanMessage)
        assert memory._messages[0].content == "Hello"
        assert isinstance(memory._messages[1], AIMessage)
        assert memory._messages[1].content == "Hi"
        assert isinstance(memory._messages[2], SystemMessage)
        assert memory._messages[2].content == "Summary"

    def test_clear(self, fake_redis, mock_summary_manager, test_session_id, test_messages):
        """Test clearing chat history."""
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        memory._messages = test_messages
        memory._save_to_redis()
        
        # Verify data exists before clearing
        assert fake_redis.get(memory._get_redis_key()) is not None
        assert len(memory._messages) > 0
        
        memory.clear()
        
        # Verify data is cleared
        assert memory._messages == []
        assert fake_redis.get(memory._get_redis_key()) is None

    def test_load_from_redis_invalid_json(self, fake_redis, mock_summary_manager, test_session_id):
        """Test handling of invalid JSON in Redis."""
        # Put invalid JSON in Redis
        fake_redis.set(f"chat_history:{test_session_id}", "invalid json data")
        
        memory = RedisChatMessageHistory(
            session_id=test_session_id,
            redis_client=fake_redis,
            summary_manager=mock_summary_manager
        )
        
        # Should handle gracefully and have empty messages
        assert memory._messages == []

    def test_max_messages_constant(self):
        """Test that MAX_MESSAGES constant is set correctly."""
        assert RedisChatMessageHistory.MAX_MESSAGES == 20

    def test_summary_batch_size_constant(self):
        """Test that SUMMARY_BATCH_SIZE constant is set correctly."""
        assert RedisChatMessageHistory.SUMMARY_BATCH_SIZE == 10