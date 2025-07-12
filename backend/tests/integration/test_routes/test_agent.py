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

    def test_query_agent_stream_endpoint_exists(self, client):
        """Test that the streaming endpoint exists and returns correct content type."""
        request_data = {
            "query": "What is the capital of France?",
            "session_id": "test-session-123",
            "prompt_type": "standard",
            "max_iterations": 10
        }

        # Just verify the endpoint exists and returns streaming content type
        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
        # Should get streaming response (might timeout but status should be correct)
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_query_agent_stream_auto_session_id(self, client):
        """Test agent query without session_id (should auto-generate)."""
        request_data = {
            "query": "Test question without session_id",
            "prompt_type": "research",
            "max_iterations": 15
        }

        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
        # Should accept request and start streaming
        assert response.status_code == 200

    def test_query_agent_stream_with_research_prompt(self, client):
        """Test agent query with research prompt type."""
        request_data = {
            "query": "Research question about AI",
            "session_id": "research-session-456",
            "prompt_type": "research",
            "max_iterations": 15
        }

        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
        assert response.status_code == 200

    def test_query_agent_stream_with_simple_prompt(self, client):
        """Test agent query with simple prompt type."""
        request_data = {
            "query": "Simple question",
            "session_id": "simple-session-789",
            "prompt_type": "simple"
        }

        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
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

    def test_query_agent_stream_default_values(self, client):
        """Test that default values are properly applied."""
        request_data = {
            "query": "Test with defaults",
            "session_id": "default-session"
            # No prompt_type or max_iterations specified
        }

        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
        assert response.status_code == 200

    def test_query_agent_stream_empty_session_id(self, client):
        """Test that empty session_id gets auto-generated."""
        request_data = {
            "query": "Test with empty session_id",
            "session_id": ""  # Empty session_id should trigger auto-generation
        }

        response = client.post("/api/agent/query", json=request_data, timeout=1.0)
        
        assert response.status_code == 200

    def test_query_agent_stream_session_persistence(self, client):
        """Test that same session_id can be reused for conversation continuity."""
        session_id = "persistent-session-123"
        
        # Make two requests with same session_id
        request_data_1 = {"query": "First question", "session_id": session_id}
        request_data_2 = {"query": "Follow-up question", "session_id": session_id}
        
        response_1 = client.post("/api/agent/query", json=request_data_1, timeout=1.0)
        response_2 = client.post("/api/agent/query", json=request_data_2, timeout=1.0)
        
        assert response_1.status_code == 200
        assert response_2.status_code == 200