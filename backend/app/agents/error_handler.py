from dataclasses import asdict
import logging
from typing import Dict, Any
from datetime import datetime

from .state_machine import AgentStateMachine, AgentState
from ..utils.error_utils import create_error_context
from .exceptions import (
    APIError,
    LLMAPIError,
    ExternalAPIError,
    TransientError,
    RateLimitError,
    AuthenticationError,
    NetworkError,
    RedisError,
    SearchAPIError,
)

logger = logging.getLogger(__name__)


class APIErrorHandler:

    def __init__(self, state_machine: AgentStateMachine):
        self.state_machine = state_machine

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        error_context = create_error_context(error)

        self.state_machine.transition_to(AgentState.ERROR, asdict(error_context))

        logger.error(
            f"API error occurred in session {self.state_machine.session_id}: "
            f"{error_context.error_type} - {error_context.message}\n"
            f"Traceback: {error_context.traceback}",
            exc_info=error if not isinstance(error, APIError) else error.__cause__,
        )

        return {
            "error": True,
            "error_type": error_context.error_type,
            "message": error_context.message,
            "is_retryable": error_context.is_retryable,
            "session_id": self.state_machine.session_id,
            "timestamp": datetime.now().isoformat(),
            "traceback": error_context.traceback,
            "original_exception": error_context.original_exception_type,
        }

    def handle_llm_error(self, error: Exception) -> Dict[str, Any]:
        """Handle LLM API specific errors with enhanced traceback preservation"""

        try:
            if "rate limit" in str(error).lower():
                api_error = RateLimitError(
                    "OpenAI rate limit exceeded",
                    "RATE_LIMIT",
                    error,
                )
            elif "authentication" in str(error).lower() or "401" in str(error):
                api_error = AuthenticationError(
                    "OpenAI authentication failed",
                    "AUTH_ERROR",
                    error,
                )
            elif "timeout" in str(error).lower() or "network" in str(error).lower():
                api_error = NetworkError(
                    "OpenAI network error",
                    "NETWORK_ERROR",
                    error,
                )
            else:
                api_error = LLMAPIError(
                    "OpenAI API error",
                    "LLM_ERROR",
                    error,
                )

            # Preserve the original traceback by raising from
            raise api_error from error

        except APIError as api_err:
            try:
                return self.handle_error(api_err)
            except Exception as e:
                # If handle_error fails, return a basic error response
                return {
                    "error": True,
                    "error_type": "LLMAPIError",
                    "message": f"Unexpected error in LLM error handling: {str(e)}",
                    "is_retryable": False,
                    "session_id": self.state_machine.session_id,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            # Fallback handling
            fallback_error = LLMAPIError(
                f"Unexpected error in LLM error handling: {str(e)}",
                "LLM_HANDLER_ERROR",
                e,
            )
            try:
                return self.handle_error(fallback_error)
            except Exception:
                # If handle_error also fails, return a basic error response
                return {
                    "error": True,
                    "error_type": "LLMAPIError",
                    "message": f"Unexpected error in LLM error handling: {str(e)}",
                    "is_retryable": False,
                    "session_id": self.state_machine.session_id,
                    "timestamp": datetime.now().isoformat(),
                }

    def handle_external_api_error(
        self, error: Exception, api_name: str = "external"
    ) -> Dict[str, Any]:
        """Handle external API errors with enhanced traceback preservation"""

        try:
            if "redis" in str(error).lower() or "connection" in str(error).lower():
                api_error = RedisError(
                    f"Redis connection error: {str(error)}",
                    "REDIS_ERROR",
                    error,
                    {"api_name": api_name},
                )
            elif "search" in str(error).lower() or "duckduckgo" in str(error).lower():
                api_error = SearchAPIError(
                    f"Search API error: {str(error)}",
                    "SEARCH_ERROR",
                    error,
                    {"api_name": api_name},
                )
            elif "timeout" in str(error).lower():
                api_error = NetworkError(
                    f"{api_name} timeout error: {str(error)}",
                    "TIMEOUT_ERROR",
                    error,
                    {"api_name": api_name},
                )
            else:
                api_error = ExternalAPIError(
                    f"{api_name} API error: {str(error)}",
                    "EXTERNAL_API_ERROR",
                    error,
                    {"api_name": api_name},
                )

            # Preserve the original traceback by raising from
            raise api_error from error

        except APIError as api_err:
            try:
                return self.handle_error(api_err)
            except Exception as e:
                # If handle_error fails, return a basic error response
                return {
                    "error": True,
                    "error_type": "ExternalAPIError",
                    "message": f"Unexpected error in external API error handling: {str(e)}",
                    "is_retryable": False,
                    "session_id": self.state_machine.session_id,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            # Fallback handling
            fallback_error = ExternalAPIError(
                f"Unexpected error in external API error handling: {str(e)}",
                "EXTERNAL_HANDLER_ERROR",
                e,
                {"api_name": api_name},
            )
            try:
                return self.handle_error(fallback_error)
            except Exception:
                # If handle_error also fails, return a basic error response
                return {
                    "error": True,
                    "error_type": "ExternalAPIError",
                    "message": f"Unexpected error in external API error handling: {str(e)}",
                    "is_retryable": False,
                    "session_id": self.state_machine.session_id,
                    "timestamp": datetime.now().isoformat(),
                }



    def can_retry(self) -> bool:
        """Check if current error state allows retry"""
        if self.state_machine.current_state != AgentState.ERROR:
            return False

        if not self.state_machine.error_context:
            return False

        return self.state_machine.error_context.get("is_retryable", False)

    def clear_error(self) -> bool:
        """Clear error state and transition to IDLE"""
        if self.state_machine.current_state == AgentState.ERROR:
            return self.state_machine.transition_to(AgentState.IDLE)
        return False
