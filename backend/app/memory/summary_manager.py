from typing import List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


class ConversationSummaryManager:
    def __init__(self, llm: ChatOpenAI, max_summary_length: int = 1000):
        self.llm = llm
        self.max_summary_length = max_summary_length

    def summarize_messages(self, messages: List[BaseMessage]) -> str:
        if not messages:
            return ""

        conversation_text = "\n".join(
            [f"{msg.__class__.__name__}: {msg.content}" for msg in messages]
        )

        prompt = f"""Summarize the following conversation in about {self.max_summary_length // 2} characters:
        
        {conversation_text}
        
        Summary:"""

        response = self.llm.invoke(prompt)
        return str(response.content)