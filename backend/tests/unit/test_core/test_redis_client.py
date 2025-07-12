import pytest
import os
from unittest.mock import patch, Mock
import redis
from app.core.redis_client import RedisClient


class TestRedisClient:
    """Test cases for RedisClient singleton."""

    def test_get_client_creates_instance(self, mock_env_vars):
        """Test that get_client creates a Redis instance."""
        with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
            with patch('app.core.redis_client.Redis') as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.from_url.return_value = mock_pool
                
                mock_redis_instance = Mock()
                mock_redis_instance.ping.return_value = True
                mock_redis_class.return_value = mock_redis_instance
                
                client = RedisClient.get_client()
                
                assert client == mock_redis_instance
                mock_pool_class.from_url.assert_called_once_with(
                    "redis://localhost:6379/0", decode_responses=True, max_connections=10
                )
                mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
                mock_redis_instance.ping.assert_called_once()

    def test_get_client_singleton_behavior(self, mock_env_vars):
        """Test that get_client returns the same instance on multiple calls."""
        with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
            with patch('app.core.redis_client.Redis') as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.from_url.return_value = mock_pool
                
                mock_redis_instance = Mock()
                mock_redis_instance.ping.return_value = True
                mock_redis_class.return_value = mock_redis_instance
                
                client1 = RedisClient.get_client()
                client2 = RedisClient.get_client()
                
                assert client1 is client2
                mock_redis_class.assert_called_once()

    def test_get_client_with_custom_redis_url(self):
        """Test client creation with custom Redis URL."""
        custom_url = "redis://custom-host:6380/1"
        with patch.dict(os.environ, {'REDIS_URL': custom_url}):
            with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
                with patch('app.core.redis_client.Redis') as mock_redis_class:
                    mock_pool = Mock()
                    mock_pool_class.from_url.return_value = mock_pool
                    
                    mock_redis_instance = Mock()
                    mock_redis_instance.ping.return_value = True
                    mock_redis_class.return_value = mock_redis_instance
                    
                    client = RedisClient.get_client()
                    
                    assert client == mock_redis_instance
                    mock_pool_class.from_url.assert_called_once_with(
                        custom_url, decode_responses=True, max_connections=10
                    )

    def test_get_client_connection_error(self, mock_env_vars):
        """Test that Redis connection error is properly handled."""
        with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
            with patch('app.core.redis_client.Redis') as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.from_url.return_value = mock_pool
                
                mock_redis_instance = Mock()
                mock_redis_instance.ping.side_effect = redis.ConnectionError("Connection failed")
                mock_redis_class.return_value = mock_redis_instance
                
                with pytest.raises(redis.ConnectionError, match="Connection failed"):
                    RedisClient.get_client()

    def test_get_client_uses_default_url(self):
        """Test that client uses default Redis URL when not specified."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
                with patch('app.core.redis_client.Redis') as mock_redis_class:
                    mock_pool = Mock()
                    mock_pool_class.from_url.return_value = mock_pool
                    
                    mock_redis_instance = Mock()
                    mock_redis_instance.ping.return_value = True
                    mock_redis_class.return_value = mock_redis_instance
                    
                    client = RedisClient.get_client()
                    
                    assert client == mock_redis_instance
                    mock_pool_class.from_url.assert_called_once_with(
                        "redis://localhost:6379/0", decode_responses=True, max_connections=10
                    )

    def test_reset_instance(self, mock_env_vars):
        """Test that singleton can be reset for testing."""
        with patch('app.core.redis_client.ConnectionPool') as mock_pool_class:
            with patch('app.core.redis_client.Redis') as mock_redis_class:
                mock_pool1 = Mock()
                mock_pool2 = Mock()
                mock_pool_class.from_url.side_effect = [mock_pool1, mock_pool2]
                
                mock_redis1 = Mock()
                mock_redis1.ping.return_value = True
                mock_redis2 = Mock()
                mock_redis2.ping.return_value = True
                mock_redis_class.side_effect = [mock_redis1, mock_redis2]
                
                # Get first instance
                client1 = RedisClient.get_client()
                assert client1 == mock_redis1
                
                # Reset singleton
                RedisClient._instance = None
                RedisClient._pool = None
                
                # Get second instance
                client2 = RedisClient.get_client()
                assert client2 == mock_redis2
                assert client1 is not client2