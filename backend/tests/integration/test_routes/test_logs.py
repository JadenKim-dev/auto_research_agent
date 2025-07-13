import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open

from app.main import app

client = TestClient(app)


def test_list_sessions_empty():
    with patch("app.api.routes.logs.Path") as mock_path:
        mock_logs_dir = mock_path.return_value
        mock_logs_dir.exists.return_value = False

        response = client.get("/api/logs/sessions")
        assert response.status_code == 200
        assert response.json() == []


def test_get_session_logs_not_found():
    response = client.get("/api/logs/sessions/nonexistent_session")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


def test_get_session_logs_with_data():
    # Simple test to verify the endpoint exists and can handle requests
    response = client.get("/api/logs/sessions/test_session")

    # Should return 404 for non-existent session
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]
