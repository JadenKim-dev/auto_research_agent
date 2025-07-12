import os
from typing import Optional
import redis
from redis import Redis, ConnectionPool
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    _instance: Optional[Redis] = None
    _pool: Optional[ConnectionPool] = None

    @classmethod
    def get_client(cls) -> Redis:
        if cls._instance is not None:
            return cls._instance

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        cls._pool = ConnectionPool.from_url(
            redis_url, decode_responses=True, max_connections=10
        )
        cls._instance = Redis(connection_pool=cls._pool)

        try:
            cls._instance.ping()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

        logger.info("Redis connection established successfully")
        return cls._instance

    @classmethod
    def close(cls):
        if cls._instance:
            cls._instance.close()
            cls._instance = None
        if cls._pool:
            cls._pool.disconnect()
            cls._pool = None
