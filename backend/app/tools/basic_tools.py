from datetime import datetime
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import math


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', 'sqrt(16)')"
    )


class DateTimeInput(BaseModel):
    format: Optional[str] = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Format string for datetime (default: YYYY-MM-DD HH:MM:SS)",
    )


class WordCountInput(BaseModel):
    text: str = Field(description="Text to count words in")
    count_type: str = Field(
        default="words", description="What to count: 'words', 'characters', or 'lines'"
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The result of the calculation as a string
    """
    try:
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "__builtins__": {},
        }

        result = eval(expression, safe_dict)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool("get_current_time", args_schema=DateTimeInput)
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.

    Args:
        format: Format string for the datetime output

    Returns:
        Current datetime formatted according to the format string
    """
    try:
        current_time = datetime.now()
        return f"Current time: {current_time.strftime(format)}"
    except Exception as e:
        return f"Error formatting time: {str(e)}"


@tool("string_reverser")
def reverse_string(text: str) -> str:
    """
    Reverse a given string. Useful for testing text manipulation.

    Args:
        text: The string to reverse

    Returns:
        The reversed string
    """
    return f"Reversed string: {text[::-1]}"


@tool("word_counter", args_schema=WordCountInput)
def count_words(text: str, count_type: str = "words") -> str:
    """
    Count words, characters, or lines in a given text.

    Args:
        text: The text to analyze
        count_type: What to count ('words', 'characters', or 'lines')

    Returns:
        The count result as a formatted string
    """
    if count_type == "words":
        count = len(text.split())
        return f"Word count: {count}"
    elif count_type == "characters":
        count = len(text)
        return f"Character count: {count}"
    elif count_type == "lines":
        count = len(text.splitlines())
        return f"Line count: {count}"
    else:
        return (
            f"Unknown count type: {count_type}. Use 'words', 'characters', or 'lines'"
        )


BASIC_TOOLS = [calculator, get_current_time, reverse_string, count_words]

__all__ = [
    "BASIC_TOOLS",
    "calculator",
    "get_current_time",
    "reverse_string",
    "count_words",
]
