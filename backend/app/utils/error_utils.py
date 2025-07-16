import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..agents.exceptions import APIError, TransientError


@dataclass
class ErrorContext:
    """Error context data structure"""
    error_type: str
    message: str
    error_code: Optional[str]
    details: Dict[str, Any]
    is_retryable: bool
    timestamp: str
    traceback: str
    original_exception_type: Optional[str] = None
    original_traceback: Optional[str] = None


def extract_traceback(error: BaseException) -> str:
    """Extract formatted traceback from exception"""
    try:
        if hasattr(error, "__traceback__") and error.__traceback__:
            return "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        else:
            return traceback.format_exc()
    except Exception:
        return f"Error extracting traceback: {str(error)}"


def create_error_context(error: Exception) -> ErrorContext:
    """Create comprehensive error context with traceback preservation"""
    
    tb_str = extract_traceback(error)
    
    if isinstance(error, APIError):
        original_tb = None
        if error.__cause__:
            original_tb = extract_traceback(error.__cause__)
        
        return ErrorContext(
            error_type=error.__class__.__name__,
            message=error.message,
            error_code=error.error_code,
            details=error.details,
            is_retryable=isinstance(error, TransientError),
            timestamp=datetime.now().isoformat(),
            traceback=tb_str,
            original_exception_type=(
                error.__cause__.__class__.__name__ if error.__cause__ else None
            ),
            original_traceback=original_tb,
        )
    else:
        return ErrorContext(
            error_type="UnhandledError",
            message=str(error),
            error_code="UNKNOWN",
            details={"original_type": error.__class__.__name__},
            is_retryable=False,
            timestamp=datetime.now().isoformat(),
            traceback=tb_str,
        )

