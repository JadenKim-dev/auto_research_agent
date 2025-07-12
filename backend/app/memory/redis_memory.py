from __future__ import annotations

import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

import redis
from langchain.memory import ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


class RedisConversationSummaryMemory(ConversationSummaryMemory):
    def __init__(
        self,
        session_id: str,
        llm: ChatOpenAI,
        redis_client: redis.Redis,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        max_summary_length: int = 1000,
        ttl: int = 86400,  # 24 hours default TTL
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            memory_key=memory_key,
            return_messages=return_messages,
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )

        self.session_id = session_id
        self.max_summary_length = max_summary_length
        self.ttl = ttl

        self.redis_client = redis_client

        self._load_from_redis()

    def _get_redis_key(self) -> str:
        return f"conversation_summary:{self.session_id}"


    def _serialize_message(self, msg) -> Dict[str, Any]:
        return {
            "type": msg.__class__.__name__,
            "content": msg.content,
            "additional_kwargs": getattr(msg, "additional_kwargs", {}),
        }

    def _serialize_messages(self) -> List[Dict[str, Any]]:
        return [self._serialize_message(msg) for msg in self.chat_memory.messages]

    def _save_to_redis(self):
        data = {
            "summary": self.moving_summary_buffer,
            "messages": self._serialize_messages(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            pipe = self.redis_client.pipeline()
            pipe.setex(self._get_redis_key(), self.ttl, json.dumps(data))
            pipe.execute()
            logger.info(f"Saved conversation summary for session {self.session_id}")
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

        self.moving_summary_buffer = saved_data.get("summary", "")

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

        self.chat_memory.messages = messages
        logger.info(f"Loaded conversation summary for session {self.session_id}")

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        super().save_context(inputs, outputs)

        if len(self.moving_summary_buffer) > self.max_summary_length:
            self._prune_summary()

        self._save_to_redis()

    def _prune_summary(self):
        prompt = f"""The following is a conversation summary that has grown too long. 
        Please provide a more concise summary that captures the key points and context in about {self.max_summary_length // 2} characters:
        
        {self.moving_summary_buffer}
        
        Concise summary:"""

        response = self.llm.invoke(prompt)
        self.moving_summary_buffer = response.content