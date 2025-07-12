import pytest
import os
from unittest.mock import patch, Mock
from app.core.llm_client import LLMClient
from langchain_openai import ChatOpenAI


class TestLLMClient:
    """Test cases for LLMClient singleton."""

    def test_get_client_creates_instance(self, mock_env_vars):
        """Test that get_client creates a ChatOpenAI instance."""
        with patch('app.core.llm_client.ChatOpenAI') as mock_chat_openai:
            mock_instance = Mock(spec=ChatOpenAI)
            mock_chat_openai.return_value = mock_instance
            
            client = LLMClient.get_client()
            
            assert client == mock_instance
            mock_chat_openai.assert_called_once()

    def test_get_client_singleton_behavior(self, mock_env_vars):
        """Test that get_client returns the same instance on multiple calls."""
        with patch('app.core.llm_client.ChatOpenAI') as mock_chat_openai:
            mock_instance = Mock(spec=ChatOpenAI)
            mock_chat_openai.return_value = mock_instance
            
            client1 = LLMClient.get_client()
            client2 = LLMClient.get_client()
            
            assert client1 is client2
            mock_chat_openai.assert_called_once()

    def test_get_client_missing_api_key(self):
        """Test that missing OPENAI_API_KEY raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is not set"):
                LLMClient.get_client()

    def test_get_client_with_custom_api_key(self):
        """Test client creation with custom API key."""
        test_api_key = "test_custom_key"
        with patch.dict(os.environ, {'OPENAI_API_KEY': test_api_key}):
            with patch('app.core.llm_client.ChatOpenAI') as mock_chat_openai:
                mock_instance = Mock(spec=ChatOpenAI)
                mock_chat_openai.return_value = mock_instance
                
                client = LLMClient.get_client()
                
                assert client == mock_instance
                # Verify that ChatOpenAI was called with the correct parameters
                call_args = mock_chat_openai.call_args
                assert call_args[1]['model'] == "gpt-4o-mini"
                assert call_args[1]['api_key'].get_secret_value() == test_api_key
                assert call_args[1]['temperature'] == 0

    def test_reset_instance(self, mock_env_vars):
        """Test that singleton can be reset for testing."""
        with patch('app.core.llm_client.ChatOpenAI') as mock_chat_openai:
            mock_instance1 = Mock(spec=ChatOpenAI)
            mock_instance2 = Mock(spec=ChatOpenAI)
            mock_chat_openai.side_effect = [mock_instance1, mock_instance2]
            
            # Get first instance
            client1 = LLMClient.get_client()
            assert client1 == mock_instance1
            
            # Reset singleton
            LLMClient._instance = None
            
            # Get second instance
            client2 = LLMClient.get_client()
            assert client2 == mock_instance2
            assert client1 is not client2