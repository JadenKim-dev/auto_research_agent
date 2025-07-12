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


class SearchInput(BaseModel):
    query: str = Field(description="Search query to find information about")
    num_results: int = Field(
        default=5, description="Number of search results to return (1-10)"
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


@tool("web_search", args_schema=SearchInput)
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information using DuckDuckGo.

    Args:
        query: The search query
        num_results: Number of results to return (1-10)

    Returns:
        Formatted search results as a string
    """
    try:
        from duckduckgo_search import DDGS
        
        num_results = max(1, min(num_results, 10))
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        if not results:
            return f"No search results found for '{query}'"
        
        result_text = f"Search results for '{query}' (showing {len(results)} results):\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "No URL")
            snippet = result.get("body", "No description available")
            
            result_text += f"{i}. **{title}**\n"
            result_text += f"   URL: {url}\n"
            result_text += f"   {snippet}\n\n"
        
        return result_text
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"


BASIC_TOOLS = [calculator, get_current_time, reverse_string, count_words, web_search]

__all__ = [
    "BASIC_TOOLS",
    "calculator",
    "get_current_time",
    "reverse_string",
    "count_words",
    "web_search",
]
