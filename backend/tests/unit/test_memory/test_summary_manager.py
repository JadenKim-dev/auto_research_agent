from unittest.mock import Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from app.memory.summary_manager import ConversationSummaryManager


class TestConversationSummaryManager:
    """Test cases for ConversationSummaryManager."""

    def test_init(self, mock_llm):
        """Test ConversationSummaryManager initialization."""
        manager = ConversationSummaryManager(llm=mock_llm, max_summary_length=500)
        
        assert manager.llm == mock_llm
        assert manager.max_summary_length == 500

    def test_init_default_max_length(self, mock_llm):
        """Test ConversationSummaryManager initialization with default max_summary_length."""
        manager = ConversationSummaryManager(llm=mock_llm)
        
        assert manager.llm == mock_llm
        assert manager.max_summary_length == 1000

    def test_summarize_messages_empty_list(self, mock_llm):
        """Test summarizing empty message list returns empty string."""
        manager = ConversationSummaryManager(llm=mock_llm)
        
        result = manager.summarize_messages([])
        
        assert result == ""
        mock_llm.invoke.assert_not_called()

    def test_summarize_messages_single_message(self, mock_llm):
        """Test summarizing a single message."""
        manager = ConversationSummaryManager(llm=mock_llm, max_summary_length=1000)
        messages: List[BaseMessage] = [HumanMessage(content="Hello, how are you?")]
        
        mock_response = Mock()
        mock_response.content = "User greeted and asked about wellbeing"
        mock_llm.invoke.return_value = mock_response
        
        result = manager.summarize_messages(messages)
        
        assert result == "User greeted and asked about wellbeing"
        mock_llm.invoke.assert_called_once()
        
        # Check that the prompt contains the message content
        call_args = mock_llm.invoke.call_args[0][0]
        assert "HumanMessage: Hello, how are you?" in call_args
        assert "500 characters" in call_args  # max_summary_length // 2

    def test_summarize_messages_multiple_messages(self, mock_llm):
        """Test summarizing multiple messages."""
        manager = ConversationSummaryManager(llm=mock_llm, max_summary_length=800)
        messages = [
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="I don't have access to weather data."),
            HumanMessage(content="Can you help me with math?"),
            AIMessage(content="Yes, I can help with math problems.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Conversation about weather and math assistance"
        mock_llm.invoke.return_value = mock_response
        
        result = manager.summarize_messages(messages)
        
        assert result == "Conversation about weather and math assistance"
        mock_llm.invoke.assert_called_once()
        
        # Check that the prompt contains all message contents
        call_args = mock_llm.invoke.call_args[0][0]
        assert "HumanMessage: What's the weather like?" in call_args
        assert "AIMessage: I don't have access to weather data." in call_args
        assert "HumanMessage: Can you help me with math?" in call_args
        assert "AIMessage: Yes, I can help with math problems." in call_args
        assert "400 characters" in call_args  # max_summary_length // 2

    def test_summarize_messages_with_system_message(self, mock_llm):
        """Test summarizing messages including SystemMessage."""
        manager = ConversationSummaryManager(llm=mock_llm)
        messages = [
            SystemMessage(content="Previous summary: User asked about help"),
            HumanMessage(content="Tell me about AI"),
            AIMessage(content="AI is artificial intelligence technology.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Discussion about AI after previous help inquiry"
        mock_llm.invoke.return_value = mock_response
        
        result = manager.summarize_messages(messages)
        
        assert result == "Discussion about AI after previous help inquiry"
        
        # Check that SystemMessage is included in the conversation text
        call_args = mock_llm.invoke.call_args[0][0]
        assert "SystemMessage: Previous summary: User asked about help" in call_args
        assert "HumanMessage: Tell me about AI" in call_args
        assert "AIMessage: AI is artificial intelligence technology." in call_args

    def test_summarize_messages_prompt_format(self, mock_llm):
        """Test that the prompt is formatted correctly."""
        manager = ConversationSummaryManager(llm=mock_llm, max_summary_length=600)
        messages: List[BaseMessage] = [HumanMessage(content="Test message")]
        
        mock_response = Mock()
        mock_response.content = "Test summary"
        mock_llm.invoke.return_value = mock_response
        
        manager.summarize_messages(messages)
        
        call_args = mock_llm.invoke.call_args[0][0]
        
        # Check prompt structure
        assert "Summarize the following conversation in about 300 characters:" in call_args
        assert "HumanMessage: Test message" in call_args
        assert "Summary:" in call_args

    def test_summarize_messages_returns_string_content(self, mock_llm):
        """Test that the method returns string content from LLM response."""
        manager = ConversationSummaryManager(llm=mock_llm)
        messages: List[BaseMessage] = [HumanMessage(content="Test")]
        
        # Test with different response object types
        mock_response = Mock()
        mock_response.content = 42  # Non-string content
        mock_llm.invoke.return_value = mock_response
        
        result = manager.summarize_messages(messages)
        
        assert result == "42"  # Should be converted to string