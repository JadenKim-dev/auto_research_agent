import pytest
from unittest.mock import Mock
from datetime import datetime

from app.agents.state_machine import AgentStateMachine, AgentState
from app.agents.error_handler import APIErrorHandler
from app.agents.exceptions import (
    LLMAPIError, ExternalAPIError, TransientError,
    RateLimitError, AuthenticationError, NetworkError,
    RedisError, SearchAPIError
)


class TestAPIErrorHandler:
    
    @pytest.fixture
    def state_machine(self):
        return AgentStateMachine("test-session")
    
    @pytest.fixture
    def error_handler(self, state_machine):
        return APIErrorHandler(state_machine)

    def test_handle_llm_rate_limit_error(self, error_handler):
        error = Exception("Rate limit exceeded for OpenAI API")
        
        result = error_handler.handle_llm_error(error)
        
        assert result["error"] is True
        assert result["error_type"] == "RateLimitError"
        assert "rate limit" in result["message"].lower()
        assert result["is_retryable"] is True
        assert error_handler.state_machine.current_state == AgentState.ERROR

    def test_handle_llm_authentication_error(self, error_handler):
        error = Exception("Authentication failed: 401 Unauthorized")
        
        result = error_handler.handle_llm_error(error)
        
        assert result["error"] is True
        assert result["error_type"] == "AuthenticationError"
        assert result["is_retryable"] is False
        assert error_handler.state_machine.current_state == AgentState.ERROR

    def test_handle_llm_network_error(self, error_handler):
        error = Exception("Network timeout occurred")
        
        result = error_handler.handle_llm_error(error)
        
        assert result["error"] is True
        assert result["error_type"] == "NetworkError"
        assert result["is_retryable"] is True

    def test_handle_llm_generic_error(self, error_handler):
        error = Exception("Some other OpenAI error")
        
        result = error_handler.handle_llm_error(error)
        
        assert result["error"] is True
        assert result["error_type"] == "LLMAPIError"
        assert result["is_retryable"] is False

    def test_handle_redis_error(self, error_handler):
        error = Exception("Redis connection failed")
        
        result = error_handler.handle_external_api_error(error, "redis")
        
        assert result["error"] is True
        assert result["error_type"] == "RedisError"
        assert "redis" in result["message"].lower()

    def test_handle_search_api_error(self, error_handler):
        error = Exception("DuckDuckGo search failed")
        
        result = error_handler.handle_external_api_error(error, "search")
        
        assert result["error"] is True
        assert result["error_type"] == "SearchAPIError"
        assert "search" in result["message"].lower()

    def test_handle_external_timeout_error(self, error_handler):
        error = Exception("Request timeout")
        
        result = error_handler.handle_external_api_error(error, "external")
        
        assert result["error"] is True
        assert result["error_type"] == "NetworkError"
        assert result["is_retryable"] is True

    def test_handle_external_generic_error(self, error_handler):
        error = Exception("Some external API error")
        
        result = error_handler.handle_external_api_error(error, "test_api")
        
        assert result["error"] is True
        assert result["error_type"] == "ExternalAPIError"
        assert "test_api" in result["message"]

    def test_handle_non_api_error(self, error_handler):
        error = ValueError("Some generic error")
        
        result = error_handler.handle_error(error)
        
        assert result["error"] is True
        assert result["error_type"] == "UnhandledError"
        assert result["is_retryable"] is False

    def test_can_retry_with_transient_error(self, error_handler):
        # Set up error state with retryable error
        transient_error = RateLimitError("Rate limit exceeded")
        error_handler.handle_error(transient_error)
        
        assert error_handler.can_retry() is True

    def test_can_retry_with_permanent_error(self, error_handler):
        # Set up error state with non-retryable error
        auth_error = AuthenticationError("Auth failed")
        error_handler.handle_error(auth_error)
        
        assert error_handler.can_retry() is False

    def test_can_retry_when_not_in_error_state(self, error_handler):
        # Agent in IDLE state
        assert error_handler.can_retry() is False

    def test_clear_error_from_error_state(self, error_handler):
        # Set up error state
        error = Exception("Test error")
        error_handler.handle_error(error)
        
        assert error_handler.state_machine.current_state == AgentState.ERROR
        
        result = error_handler.clear_error()
        
        assert result is True
        assert error_handler.state_machine.current_state == AgentState.IDLE

    def test_clear_error_when_not_in_error_state(self, error_handler):
        # Agent in IDLE state
        result = error_handler.clear_error()
        
        assert result is False
        assert error_handler.state_machine.current_state == AgentState.IDLE

    def test_error_context_structure(self, error_handler):
        error = RateLimitError("Rate limit exceeded", "RATE_LIMIT", None, {"retry_after": 60})
        
        result = error_handler.handle_error(error)
        
        assert "session_id" in result
        assert "timestamp" in result
        assert result["session_id"] == "test-session"
        
        # Check state machine error context
        error_context = error_handler.state_machine.error_context
        assert error_context["error_type"] == "RateLimitError"
        assert error_context["error_code"] == "RATE_LIMIT"
        assert error_context["details"]["retry_after"] == 60

    def test_state_transition_on_error(self, error_handler, state_machine):
        # Start with RUNNING state
        state_machine.transition_to(AgentState.RUNNING)
        
        error = Exception("Test error")
        error_handler.handle_error(error)
        
        assert state_machine.current_state == AgentState.ERROR
        assert len(state_machine.state_history) == 2  # IDLE->RUNNING, RUNNING->ERROR

    def test_traceback_included_in_response(self, error_handler):
        # Test that traceback is included in the response
        try:
            raise ValueError("Test error with traceback")
        except ValueError as e:
            result = error_handler.handle_error(e)
            
            assert "traceback" in result
            assert result["traceback"] is not None
            assert "ValueError: Test error with traceback" in result["traceback"]

    def test_original_exception_tracking(self, error_handler):
        # Test that original exception is tracked when handling APIError
        original_error = ValueError("Original error")
        api_error = LLMAPIError("Wrapped API error", "WRAPPED", original_error)
        
        result = error_handler.handle_error(api_error)
        
        assert result["original_exception"] == "ValueError"
        assert "original_traceback" in error_handler.state_machine.error_context

    def test_handle_llm_error_fallback_handling(self, error_handler):
        # Mock handle_error to raise an exception
        original_handle_error = error_handler.handle_error
        def mock_handle_error(error):
            if isinstance(error, RateLimitError):
                raise RuntimeError("Mock error in handle_error")
            return original_handle_error(error)
        
        error_handler.handle_error = mock_handle_error
        
        # Test fallback handling when exception occurs in try block
        error = Exception("Rate limit exceeded")
        result = error_handler.handle_llm_error(error)
        
        assert result["error_type"] == "LLMAPIError"
        assert "Unexpected error in LLM error handling" in result["message"]

    def test_handle_external_api_error_fallback_handling(self, error_handler):
        # Mock handle_error to raise an exception
        original_handle_error = error_handler.handle_error
        def mock_handle_error(error):
            if isinstance(error, RedisError):
                raise RuntimeError("Mock error in handle_error")
            return original_handle_error(error)
        
        error_handler.handle_error = mock_handle_error
        
        # Test fallback handling when exception occurs in try block
        error = Exception("Redis connection failed")
        result = error_handler.handle_external_api_error(error, "redis")
        
        assert result["error_type"] == "ExternalAPIError"
        assert "Unexpected error in external API error handling" in result["message"]

    def test_can_retry_with_no_error_context(self, error_handler):
        # Force ERROR state without error_context
        error_handler.state_machine.current_state = AgentState.ERROR
        error_handler.state_machine.error_context = None
        
        assert error_handler.can_retry() is False

    def test_error_message_includes_api_name(self, error_handler):
        # Test that api_name is included in error messages
        error = Exception("Some API error")
        
        result = error_handler.handle_external_api_error(error, "custom_api")
        
        assert "custom_api" in result["message"]
        assert error_handler.state_machine.error_context["details"]["api_name"] == "custom_api"

    def test_connection_error_handled_as_redis(self, error_handler):
        # Test that "connection" errors are treated as Redis errors
        error = Exception("Connection refused")
        
        result = error_handler.handle_external_api_error(error, "some_api")
        
        assert result["error_type"] == "RedisError"
        assert "connection" in result["message"].lower()