import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Flask
import os
import requests
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api import api_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_start_pea_session_missing_api_key(client):
    """Test starting PEA session with missing API key returns 500"""
    with patch.dict(os.environ, {}, clear=True):
        response = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )
        assert response.status_code == 500
        assert "GOOGLE_API_KEY or OPENAI_API_KEY not set" in response.json["error"]


def test_pea_chat_invalid_session(client):
    """Test PEA chat with invalid session ID returns 400"""
    response = client.post(
        "/api/pea/chat",
        data=json.dumps({"session_id": "invalid", "message": "Hello"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Invalid session ID" in response.json["error"]


def test_finalize_prompt_invalid_session(client):
    """Test finalizing prompt with invalid session ID returns 400"""
    response = client.post(
        "/api/pea/finalize",
        data=json.dumps({"session_id": "invalid"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Invalid session ID" in response.json["error"]


def test_pea_chat_network_error(client, mocker):
    """Test network error during PEA chat returns 500"""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy"}),
        patch("api._generate_chat_response") as mock_generate,
    ):
        # First call returns successfully for session creation
        mock_generate.side_effect = [
            "Session created",
            requests.ConnectionError("Network down"),
        ]

        # Start session
        start_resp = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json["session_id"]

        # Now send chat message that causes network error
        response = client.post(
            "/api/pea/chat",
            data=json.dumps({"session_id": session_id, "message": "Hello"}),
            content_type="application/json",
        )

        assert response.status_code == 500
        assert "Request error: Network down" in response.json["error"]


def test_finalize_prompt_timeout(client, mocker):
    """Test timeout during finalize prompt returns 500"""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy"}),
        patch("api._generate_chat_response") as mock_generate,
    ):
        # First call returns successfully for session creation
        mock_generate.side_effect = ["Session created", TimeoutError("API timeout")]

        # Start session
        start_resp = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json["session_id"]

        # Now finalize the prompt that causes timeout
        response = client.post(
            "/api/pea/finalize",
            data=json.dumps({"session_id": session_id}),
            content_type="application/json",
        )

        assert response.status_code == 500
        assert "Timeout error: API timeout" in response.json["error"]


def test_pea_session_value_error(client, mocker):
    """Test ValueError during PEA session returns 500"""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy"}),
        patch("api._generate_chat_response") as mock_generate,
    ):
        mock_generate.side_effect = ValueError("Invalid parameter")

        response = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )

        assert response.status_code == 500
        assert "Value error" in response.json["error"]


def test_pea_session_key_error(client, mocker):
    """Test KeyError during PEA session returns 500"""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy"}),
        patch("api._generate_chat_response") as mock_generate,
    ):
        mock_generate.side_effect = KeyError("missing_key")

        response = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )

        assert response.status_code == 500
        assert "Key error" in response.json["error"]


def test_pea_chat_value_error(client, mocker):
    """Test ValueError during PEA chat returns 500"""
    with (
        patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy"}),
        patch("api._generate_chat_response") as mock_generate,
    ):
        # First call returns successfully for session creation
        mock_generate.side_effect = [
            "Session created",  # For session creation
            ValueError("Invalid parameter"),  # For the chat step
        ]

        # Start session
        start_resp = client.post(
            "/api/pea/start",
            data=json.dumps({"initial_request": "Test request"}),
            content_type="application/json",
        )
        assert start_resp.status_code == 200
        session_id = start_resp.json["session_id"]

        # Now send chat message that causes ValueError
        response = client.post(
            "/api/pea/chat",
            data=json.dumps({"session_id": session_id, "message": "Hello"}),
            content_type="application/json",
        )

        assert response.status_code == 500
        assert "Value error" in response.json["error"]


def test_debug_route(client):
    """Test debug route returns expected response"""
    response = client.get("/api/pea/debug_test")
    assert response.status_code == 200
    assert "Debug PEA route hit" in response.json["message"]
