import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestAgentRoutes:
    """Integration tests for agent API routes."""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI app."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200

    def test_query_agent_stream_missing_query(self, client):
        """Test that missing query parameter returns validation error."""
        request_data = {
            "session_id": "error-session"
            # Missing required 'query' field
        }
        
        response = client.post("/api/agent/query", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_query_agent_stream_invalid_prompt_type(self, client):
        """Test that invalid prompt_type returns validation error."""
        request_data = {
            "query": "Test question",
            "session_id": "error-session",
            "prompt_type": "invalid_type"  # Invalid prompt type
        }
        
        response = client.post("/api/agent/query", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_query_agent_stream_invalid_max_iterations(self, client):
        """Test that invalid max_iterations returns validation error."""
        request_data = {
            "query": "Test question",
            "session_id": "error-session",
            "max_iterations": -1  # Invalid negative value
        }
        
        response = client.post("/api/agent/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
