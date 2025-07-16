import pytest
import traceback
from datetime import datetime
from unittest.mock import patch

from app.utils.error_utils import extract_traceback, create_error_context, ErrorContext
from app.agents.exceptions import (
    APIError, TransientError, RateLimitError, AuthenticationError
)


class TestErrorUtils:
    
    def test_extract_traceback_with_exception(self):
        # Test extracting traceback from an exception with traceback
        try:
            raise ValueError("Test error")
        except ValueError as e:
            tb_str = extract_traceback(e)
            
            assert "ValueError: Test error" in tb_str
            assert "raise ValueError" in tb_str
            assert "Traceback (most recent call last):" in tb_str
    
    def test_extract_traceback_without_traceback(self):
        # Test extracting traceback from exception without __traceback__
        error = ValueError("Test error")
        error.__traceback__ = None
        
        with patch('traceback.format_exc', return_value="Mocked traceback"):
            tb_str = extract_traceback(error)
            
            assert tb_str == "Mocked traceback"
    
    def test_extract_traceback_error_handling(self):
        # Test error handling in extract_traceback
        error = Exception("Test error")
        
        # Mock to raise an exception
        with patch('traceback.format_exception', side_effect=Exception("Format error")):
            tb_str = extract_traceback(error)
            
            assert tb_str == "Error extracting traceback: Test error"
    
    def test_create_error_context_for_api_error(self):
        # Test creating error context for APIError
        try:
            raise ValueError("Original error")
        except ValueError as original_error:
            api_error = RateLimitError(
                "Rate limit exceeded",
                "RATE_LIMIT",
                original_error,
                {"retry_after": 60}
            )
        
        context = create_error_context(api_error)
        
        assert isinstance(context, ErrorContext)
        assert context.error_type == "RateLimitError"
        assert context.message == "Rate limit exceeded"
        assert context.error_code == "RATE_LIMIT"
        assert context.details == {"retry_after": 60}
        assert context.is_retryable is True
        assert context.original_exception_type == "ValueError"
        assert context.original_traceback is not None
        assert "ValueError: Original error" in context.original_traceback
    
    def test_create_error_context_for_non_retryable_api_error(self):
        # Test creating error context for non-retryable APIError
        auth_error = AuthenticationError(
            "Authentication failed",
            "AUTH_ERROR",
            None,
            {"status_code": 401}
        )
        
        context = create_error_context(auth_error)
        
        assert context.is_retryable is False
        assert context.original_exception_type is None
        assert context.original_traceback is None
    
    def test_create_error_context_for_generic_exception(self):
        # Test creating error context for generic exception
        error = ValueError("Generic error")
        
        context = create_error_context(error)
        
        assert isinstance(context, ErrorContext)
        assert context.error_type == "UnhandledError"
        assert context.message == "Generic error"
        assert context.error_code == "UNKNOWN"
        assert context.details == {"original_type": "ValueError"}
        assert context.is_retryable is False
        assert context.original_exception_type is None
        assert context.original_traceback is None
    
    def test_error_context_timestamp_format(self):
        # Test that timestamp is in ISO format
        error = ValueError("Test error")
        
        context = create_error_context(error)
        
        # Verify timestamp can be parsed
        timestamp = datetime.fromisoformat(context.timestamp)
        assert isinstance(timestamp, datetime)
    
    def test_error_context_dataclass_fields(self):
        # Test ErrorContext dataclass structure
        context = ErrorContext(
            error_type="TestError",
            message="Test message",
            error_code="TEST_CODE",
            details={"key": "value"},
            is_retryable=True,
            timestamp="2024-01-01T00:00:00",
            traceback="Test traceback",
            original_exception_type="OriginalError",
            original_traceback="Original traceback"
        )
        
        assert context.error_type == "TestError"
        assert context.message == "Test message"
        assert context.error_code == "TEST_CODE"
        assert context.details == {"key": "value"}
        assert context.is_retryable is True
        assert context.timestamp == "2024-01-01T00:00:00"
        assert context.traceback == "Test traceback"
        assert context.original_exception_type == "OriginalError"
        assert context.original_traceback == "Original traceback"
    
    def test_error_context_optional_fields(self):
        # Test ErrorContext with optional fields as None
        context = ErrorContext(
            error_type="TestError",
            message="Test message",
            error_code=None,
            details={},
            is_retryable=False,
            timestamp="2024-01-01T00:00:00",
            traceback="Test traceback"
        )
        
        assert context.error_code is None
        assert context.original_exception_type is None
        assert context.original_traceback is None