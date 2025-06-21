import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app import app
from api import api_bp, pea_conversations

# Set test environment variables
os.environ["GOOGLE_API_KEY"] = "test_api_key"
os.environ["ENV_FILE"] = ".env.test"


@pytest.fixture(autouse=True)
def clear_conversations():
    """Clear the conversation store before and after each test."""
    pea_conversations.clear()
    yield
    pea_conversations.clear()


@pytest.fixture
def client():
    """Create test client with API blueprint."""
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_genai():
    """Mock Google GenerativeModel for all tests."""
    with patch("api.genai.GenerativeModel") as mock_generative_model:
        mock_model_instance = MagicMock()
        mock_generative_model.return_value = mock_model_instance
        mock_response = MagicMock()
        mock_response.text = (
            "<optimized_prompt>Test optimized prompt</optimized_prompt>"
        )
        mock_model_instance.generate_content.return_value = mock_response
        yield mock_model_instance


def test_optimize_prompt(client, mock_genai):
    """Test /api/optimize-prompt endpoint with proper API key handling"""
    response = client.post(
        "/api/optimize-prompt",
        json={"request": "Test request", "provider": "google"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "optimized_prompt" in data
    assert (
        data["optimized_prompt"]
        == "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    )


def test_start_pea_session(client, mock_genai):
    """Test /api/pea/start endpoint with session initialization"""
    response = client.post(
        "/api/pea/start", json={"initial_request": "Test initial request"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "session_id" in data
    assert "response" in data
    session_id = data["session_id"]
    assert session_id in pea_conversations


def test_pea_chat(client, mock_genai):
    """Test /api/pea/chat endpoint with session continuation"""
    # Start a session
    start_response = client.post(
        "/api/pea/start", json={"initial_request": "Test initial request"}
    )
    start_data = start_response.get_json()
    session_id = start_data["session_id"]

    # Send chat message
    response = client.post(
        "/api/pea/chat", json={"session_id": session_id, "message": "Test message"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "response" in data


def test_finalize_prompt(client, mock_genai):
    """Test /api/pea/finalize endpoint with prompt generation"""
    # Start a session
    start_response = client.post(
        "/api/pea/start", json={"initial_request": "Test initial request"}
    )
    start_data = start_response.get_json()
    session_id = start_data["session_id"]

    # Finalize prompt
    response = client.post("/api/pea/finalize", json={"session_id": session_id})
    assert response.status_code == 200
    data = response.get_json()
    assert "final_prompt" in data
    assert session_id not in pea_conversations  # Session should be cleared


@patch("api.genai.GenerativeModel")
def test_timeout_error(mock_GenerativeModel, client):
    """Test timeout error handling"""
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_model_instance.generate_content.side_effect = TimeoutError(
        "Test timeout error"
    )

    response = client.post(
        "/api/optimize-prompt",
        json={"request": "Test request", "provider": "google"},
    )
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
    assert "Request timed out" in data["error"]


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_optimize_prompt_success(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    mock_model_instance.generate_content.return_value = mock_response

    response = client.post("/api/optimize-prompt", json={"request": "test request"})

    assert response.status_code == 200
    assert response.json == {
        "optimized_prompt": "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    }
    mock_getenv.assert_called_once_with("GOOGLE_API_KEY")
    mock_GenerativeModel.assert_called_once()
    call_args = mock_GenerativeModel.call_args
    assert call_args.kwargs["model_name"] == "gemini-pro"
    # Safety settings are now handled differently in our code
    mock_model_instance.generate_content.assert_called_once()


@patch("api.os.getenv")
def test_optimize_prompt_no_api_key(mock_getenv, client):
    mock_getenv.return_value = None

    response = client.post("/api/optimize-prompt", json={"request": "test request"})

    assert response.status_code == 500
    assert "API key not set" in response.json["error"]
    mock_getenv.assert_called_once_with("GOOGLE_API_KEY")


def test_optimize_prompt_no_request(client):
    response = client.post("/api/optimize-prompt", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}


def test_optimize_prompt_empty_request(client):
    response = client.post("/api/optimize-prompt", json={"request": ""})
    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_optimize_prompt_api_error(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_model_instance.generate_content.side_effect = Exception("API Error")

    response = client.post("/api/optimize-prompt", json={"request": "test request"})

    assert response.status_code == 500
    assert "API Error" in response.json["error"]
    mock_getenv.assert_called_once_with("GOOGLE_API_KEY")
    mock_GenerativeModel.assert_called_once()
    mock_model_instance.generate_content.assert_called_once()


# PEA Tests
@pytest.fixture(autouse=True)
def clear_conversations():
    """Clear the conversation store before and after each test."""
    pea_conversations.clear()
    yield
    pea_conversations.clear()


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_start_pea_session_success(mock_getenv, mock_GenerativeModel, client):
    # Setup mocks
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Hello! I'm PEA. Let me help you refine your prompt."
    mock_model_instance.generate_content.return_value = mock_response

    # Test request
    response = client.post(
        "/api/pea/start", json={"initial_request": "Help me create a prompt"}
    )

    # Assertions
    assert response.status_code == 200
    assert "session_id" in response.json
    assert "response" in response.json
    assert response.json["response"] == mock_response.text

    # Verify conversation was stored
    session_id = response.json["session_id"]
    assert session_id in pea_conversations
    # After the initial turn, history should contain 3 messages: system, user, model
    assert len(pea_conversations[session_id]) == 3


def test_start_pea_session_no_request(client):
    response = client.post("/api/pea/start", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No initial request provided"}


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_pea_chat_success(mock_getenv, mock_GenerativeModel, client):
    # First start a session
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Initial response"
    mock_model_instance.generate_content.return_value = mock_response

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me"})
    session_id = start_response.json["session_id"]

    # Then test chat endpoint
    mock_response.text = "Follow-up response"
    chat_response = client.post(
        "/api/pea/chat",
        json={"session_id": session_id, "message": "My follow-up question"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json == {"response": "Follow-up response"}
    # After the initial 3 messages, adding a user and model message makes it 5
    assert len(pea_conversations[session_id]) == 5


def test_pea_chat_invalid_session(client):
    response = client.post(
        "/api/pea/chat",
        json={"session_id": "invalid_session_id", "message": "test message"},
    )
    assert response.status_code == 400
    assert response.json == {"error": "Invalid session ID"}


def test_pea_chat_missing_data(client):
    response = client.post("/api/pea/chat", json={})
    assert response.status_code == 400
    assert response.json == {"error": "Missing session_id or message"}


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_finalize_prompt_success(mock_getenv, mock_GenerativeModel, client):
    # First start a session
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Initial response"
    mock_model_instance.generate_content.return_value = mock_response

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me"})
    session_id = start_response.json["session_id"]

    # Then test finalize endpoint
    mock_response.text = "<optimized_prompt>Final XML prompt</optimized_prompt>"
    finalize_response = client.post(
        "/api/pea/finalize", json={"session_id": session_id}
    )

    assert finalize_response.status_code == 200
    assert finalize_response.json == {
        "final_prompt": "<optimized_prompt>Final XML prompt</optimized_prompt>"
    }
    assert (
        session_id not in pea_conversations
    )  # Session should be cleared after finalization


def test_finalize_prompt_invalid_session(client):
    response = client.post(
        "/api/pea/finalize", json={"session_id": "invalid_session_id"}
    )
    assert response.status_code == 400
    assert response.json == {"error": "Invalid session ID"}


def test_finalize_prompt_missing_session_id(client):
    response = client.post("/api/pea/finalize", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No session_id provided"}


@patch("api.genai.GenerativeModel")
@patch("api.os.getenv")
def test_pea_api_error(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = "fake_api_key"
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_model_instance.generate_content.side_effect = Exception("API Error")

    # Test API error during PEA start
    response = client.post("/api/pea/start", json={"initial_request": "Help me"})
    assert response.status_code == 500
    assert "API Error" in response.json["error"]

    # Test API error during PEA chat (need a valid session first)
    # Start a session successfully first
    mock_response_start = MagicMock()
    mock_response_start.text = "Initial PEA response"
    mock_model_instance.generate_content.side_effect = [
        mock_response_start,
        Exception("API Error"),
    ]  # First call succeeds, second fails

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me"})
    session_id = start_response.json["session_id"]

    # Then test chat endpoint with API error
    chat_response = client.post(
        "/api/pea/chat",
        json={"session_id": session_id, "message": "My follow-up question"},
    )

    assert chat_response.status_code == 500
    assert "API Error" in chat_response.json["error"]

    # Test API error during PEA finalize (need a valid session and some history)
    # Start a session and add a chat turn successfully first
    mock_response_chat = MagicMock()
    mock_response_chat.text = "Chat response"
    mock_model_instance.generate_content.side_effect = [
        mock_response_start,
        mock_response_chat,
        Exception("API Error"),
    ]  # First two succeed, third fails

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me"})
    session_id = start_response.json["session_id"]
    client.post("/api/pea/chat", json={"session_id": session_id, "message": "Turn 1"})

    # Then test finalize endpoint with API error
    finalize_response = client.post(
        "/api/pea/finalize", json={"session_id": session_id}
    )

    assert finalize_response.status_code == 500
    assert "API Error" in finalize_response.json["error"]
    # Session should NOT be cleared if finalize fails due to API error
    assert session_id in pea_conversations


def test_handle_bad_request(client):
    """Test the BadRequest error handler for invalid JSON."""
    # Send non-JSON data to a JSON-only endpoint
    response = client.post("/api/optimize-prompt", data="not json")
    assert response.status_code == 415
    assert response.json == {
        "error": "Unsupported Media Type: Must be application/json"
    }


def test_handle_unsupported_media_type(client):
    """Test the UnsupportedMediaType error handler."""
    # Send data with wrong Content-Type
    response = client.post(
        "/api/optimize-prompt",
        data={"request": "test"},
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 415
    assert response.json == {
        "error": "Unsupported Media Type: Must be application/json"
    }
