import os
from typing import Optional
import logging

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLMClient:
    _instance: Optional[ChatOpenAI] = None

    @classmethod
    def get_client(cls) -> ChatOpenAI:
        """
        Get or create a ChatOpenAI client instance.

        Returns:
            ChatOpenAI instance
        """
        if cls._instance is not None:
            return cls._instance

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        cls._instance = ChatOpenAI(
            model="gpt-4o-mini", api_key=SecretStr(api_key), temperature=0
        )
        return cls._instance
