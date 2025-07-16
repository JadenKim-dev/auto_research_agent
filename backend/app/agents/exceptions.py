from typing import Optional, Dict, Any


class APIError(Exception):
    """Base API error class"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.__cause__ = cause
        self.details = details or {}


class LLMAPIError(APIError):
    """OpenAI/LLM API related errors"""

    pass


class ExternalAPIError(APIError):
    """External API errors (DuckDuckGo, Redis, etc.)"""

    pass


class TransientError(APIError):
    """Transient errors that can be retried"""

    pass


class RateLimitError(TransientError):
    """API rate limit exceeded"""

    pass


class AuthenticationError(APIError):
    """API authentication failed"""

    pass


class NetworkError(TransientError):
    """Network connectivity issues"""

    pass


class RedisError(ExternalAPIError):
    """Redis connection or operation errors"""

    pass


class SearchAPIError(ExternalAPIError):
    """DuckDuckGo search API errors"""

    pass
