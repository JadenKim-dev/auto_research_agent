from __future__ import annotations

import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import logging
from .summary_manager import ConversationSummaryManager

logger = logging.getLogger(__name__)


class RedisChatMessageHistory(BaseChatMessageHistory):
    MAX_MESSAGES = 20
    SUMMARY_BATCH_SIZE = 10

    def __init__(
        self,
        session_id: str,
        redis_client: redis.Redis,
        summary_manager: ConversationSummaryManager,
        ttl: int = 86400,  # 24 hours default TTL
    ):
        self.session_id = session_id
        self.redis_client = redis_client
        self.ttl = ttl
        self.summary_manager = summary_manager
        self._messages: List[BaseMessage] = []
        self._load_from_redis()

    def _get_redis_key(self) -> str:
        return f"chat_history:{self.session_id}"

    def _serialize_message(self, msg) -> Dict[str, Any]:
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        }

    def _serialize_messages(self) -> List[Dict[str, Any]]:
        return [self._serialize_message(msg) for msg in self._messages]

    def _save_to_redis(self):
        data = {
            "messages": self._serialize_messages(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            pipe = self.redis_client.pipeline()
            pipe.setex(self._get_redis_key(), self.ttl, json.dumps(data))
            pipe.execute()
            logger.info(f"Saved chat history for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to save to Redis: {str(e)}")

    def _load_from_redis(self):
        try:
            saved_data = self.redis_client.get(self._get_redis_key())
        except Exception as e:
            logger.error(f"Failed to load from Redis: {str(e)}")
            return

        if not saved_data:
            return

        try:
            saved_data = json.loads(str(saved_data))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to parse saved data: {str(e)}")
            return

        messages = []
        for msg_data in saved_data.get("messages", []):
            if msg_data["type"] == "HumanMessage":
                messages.append(
                    HumanMessage(
                        content=msg_data["content"],
                        additional_kwargs=msg_data.get("additional_kwargs", {}),
                    )
                )
            elif msg_data["type"] == "AIMessage":
                messages.append(
                    AIMessage(
                        content=msg_data["content"],
                        additional_kwargs=msg_data.get("additional_kwargs", {}),
                    )
                )
            elif msg_data["type"] == "SystemMessage":
                messages.append(
                    SystemMessage(
                        content=msg_data["content"],
                        additional_kwargs=msg_data.get("additional_kwargs", {}),
                    )
                )

        self._messages = messages
        logger.info(f"Loaded chat history for session {self.session_id}")

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._check_and_summarize()
        self._save_to_redis()

    def _check_and_summarize(self) -> None:
        if len(self._messages) <= self.MAX_MESSAGES:
            return

        messages_to_summarize = self._messages[: self.SUMMARY_BATCH_SIZE]
        summary = self.summary_manager.summarize_messages(messages_to_summarize)

        summary_message = SystemMessage(content=f"Summary: {summary}")
        self._messages = [summary_message] + self._messages[self.SUMMARY_BATCH_SIZE :]

        logger.info(
            f"Summarized {self.SUMMARY_BATCH_SIZE} messages for session {self.session_id}"
        )

    def clear(self) -> None:
        self._messages = []
        try:
            self.redis_client.delete(self._get_redis_key())
            logger.info(f"Cleared chat history for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to clear from Redis: {str(e)}")
