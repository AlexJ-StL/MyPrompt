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


def test_openai_provider_success(client):
    """Test successful OpenAI provider integration"""
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}),
        patch("api.requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OpenAI response"}}]
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": "Test", "provider": "openai"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["optimized_prompt"] == "OpenAI response"


def test_anthropic_provider_success(client):
    """Test successful Anthropic provider integration"""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}),
        patch("api.requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Anthropic response"}]}
        mock_post.return_value = mock_response

        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": "Test", "provider": "anthropic"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["optimized_prompt"] == "Anthropic response"


def test_openai_fallback_key(client):
    """Test fallback to OPENAI_API_KEY for OpenAI-compatible providers"""
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}),
        patch("api.requests.post") as mock_post,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Fallback response"}}]
        }
        mock_post.return_value = mock_response

        # Test with OpenRouter which should fallback to OPENAI_API_KEY
        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": "Test", "provider": "openrouter"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["optimized_prompt"] == "Fallback response"


def test_invalid_provider(client):
    """Test invalid provider returns 400 error"""
    response = client.post(
        "/api/optimize-prompt",
        data=json.dumps({"request": "Test", "provider": "invalid"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Invalid provider" in response.json["error"]


def test_no_key_provider(client):
    """Test provider that doesn't require API key"""
    with patch.dict(os.environ, {}), patch("api.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Ollama response"}}]
        }
        mock_post.return_value = mock_response

        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": "Test", "provider": "ollama"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json["optimized_prompt"] == "Ollama response"
