import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from flask import Flask, g

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Now import from the project
from providers.provider_base import ProviderBase

# Set test environment variables BEFORE any imports that might check them
os.environ["GOOGLE_API_KEY"] = "test_key"
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["ENV_FILE"] = ".env.test"

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app_factory import create_app
from api import api_bp, pea_conversations, get_providers_registry
from providers.provider_base import ProviderBase


@pytest.fixture(autouse=True)
def clear_conversations():
    """Clear the conversation store before and after each test."""
    pea_conversations.clear()
    yield
    pea_conversations.clear()


@pytest.fixture
def app():
    """Create test app."""
    # Create mock providers for testing
    mock_provider = MagicMock(spec=ProviderBase)
    mock_provider.name = "google"
    mock_provider.key_env_var = "GOOGLE_API_KEY"
    mock_provider.is_api_key_valid.return_value = True
    mock_provider.api_key = "test_key"
    
    # Create a MagicMock for generate_content that returns a string
    mock_generate_content = MagicMock()
    mock_generate_content.return_value = "Test PEA response"
    mock_provider.generate_content = mock_generate_content
    
    # Create mock registry with mock provider
    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_registry.available_providers.return_value = [mock_provider]
    
    # Store references for test access
    app = create_app(provider_registry_instance=mock_registry)
    app.config["TESTING"] = True
    app.config["MOCK_PROVIDER"] = mock_provider
    app.config["MOCK_REGISTRY"] = mock_registry
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    with app.test_client() as client:
        yield client





def test_optimize_prompt(client, mock_providers):
    """Test /api/optimize-prompt endpoint with provider integration"""
    test_request = "Test request"
    test_response = "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    
    # Unpack the mock provider and registry
    mock_provider, mock_registry = mock_providers
    mock_provider.generate_content.return_value = test_response
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
        response = client.post(
            "/api/optimize-prompt",
            json={"request": test_request, "provider": "google"}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "optimized_prompt" in data
        assert data["optimized_prompt"] == test_response
        # Updated to match the actual call in the code
        expected_messages = [{"role": "user", "content": test_request}]
        mock_provider.generate_content.assert_called_once_with(expected_messages)


def test_start_pea_session(client):
    """
    Test /api/pea/start endpoint with session initialization.
    
    This test simulates the entire workflow of starting a PEA session,
    from the initial API call through conversation storage.
    """
    test_request = "Test initial request"
    test_response = "Test PEA response"
    
    # Get access to the mock provider and registry from the app config
    mock_provider = client.application.config["MOCK_PROVIDER"]
    mock_registry = client.application.config["MOCK_REGISTRY"]
    
    # Ensure the provider knows it has a valid API key
    mock_provider.is_api_key_valid.return_value = True
    # Set the api_key attribute directly
    mock_provider.api_key = "test_key"
    
    # Configure the mock provider to return a string (not a coroutine)
    mock_provider.generate_content.return_value = test_response
    # Ensure the provider knows it has a valid API key
    mock_provider.is_api_key_valid.return_value = True
    # Set the api_key attribute directly to bypass environment lookup
    mock_provider.api_key = "test_key"
    # Mock the generate_content method to return our test response
    mock_provider.generate_content.return_value = test_response
    
    # Set a valid API key in the environment
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key", "OPENAI_API_KEY": "test_key"}):
        response = client.post(
            "/api/pea/start",
            json={"initial_request": test_request, "provider": "google"}
        )
        
        # Print response if there's an error
        if response.status_code != 200:
            print(f"Error status: {response.status_code}")
            error_data = response.get_json()
            if error_data:
                print(f"Error response: {error_data}")
        
        # Assertions
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.get_json()
        assert "session_id" in data, "session_id not in response"
        assert "response" in data, "response not in response"
        assert data["response"] == test_response, f"Expected '{test_response}', got '{data['response']}'"
        
        # Verify session was stored correctly
        session_id = data["session_id"]
        assert session_id in pea_conversations, f"Session {session_id} not stored in pea_conversations"
        
        # Check that the conversation history has the expected messages
        history = pea_conversations[session_id]["history"]
        assert len(history) == 3, f"Expected 3 messages, got {len(history)}"
        assert history[0]["role"] == "system", f"First message role should be 'system', got {history[0]['role']}"
        assert history[1]["role"] == "user", f"Second message role should be 'user', got {history[1]['role']}"
        assert history[2]["role"] == "model", f"Third message role should be 'model', got {history[2]['role']}"
        
            # Verify the provider was called correctly
            # The generate_content should be called with the full conversation history
            # First with system and user messages to get the initial response
            first_call_args = mock_provider.generate_content.call_args_list[0][0][0]
            assert len(first_call_args) == 2, f"Expected 2 messages, got {len(first_call_args)}"
            assert first_call_args[0]["role"] == "system"
            assert first_call_args[1]["role"] == "user"
            assert first_call_args[1]["content"] == test_request
            
            # The response should be added to the conversation history
            assert len(history) == 3
            assert history[2]["role"] == "model"
            assert history[2]["content"] == test_response


def test_pea_chat(client, mock_providers):
    """Test /api/pea/chat endpoint with session continuation"""
    test_request = "Test initial request"
    test_message = "Test message"
    test_response = "Test PEA response"
    
    # Unpack the mock provider and registry
    mock_provider, mock_registry = mock_providers
    # First call is for starting the session, second for the chat
    mock_provider.generate_content.side_effect = [test_response, test_response]
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
        # Start a session
        start_response = client.post(
            "/api/pea/start",
            json={"initial_request": test_request, "provider": "google"}
        )
        session_id = start_response.json["session_id"]

        # Send chat message
        response = client.post(
            "/api/pea/chat",
            json={"session_id": session_id, "message": test_message, "provider": "google"}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "response" in data
        assert data["response"] == test_response
        assert mock_provider.generate_content.call_count == 2


def test_finalize_prompt(client, mock_providers):
    """Test /api/pea/finalize endpoint with prompt generation"""
    test_request = "Test initial request"
    start_response = "Initial response"
    finalize_response = "<optimized_prompt>Final XML</optimized_prompt>"
    
    # Unpack the mock provider and registry
    mock_provider, mock_registry = mock_providers
    # First call for start, second for finalize
    mock_provider.generate_content.side_effect = [start_response, finalize_response]
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
        # Start a session
        start_response = client.post(
            "/api/pea/start",
            json={"initial_request": test_request, "provider": "google"}
        )
        session_id = start_response.json["session_id"]

        # Finalize prompt
        response = client.post("/api/pea/finalize", json={"session_id": session_id, "provider": "google"})
        
        assert response.status_code == 200
        data = response.get_json()
        assert "final_prompt" in data
        assert data["final_prompt"] == finalize_response
        assert session_id not in pea_conversations
        assert mock_provider.generate_content.call_count == 2


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


@patch("api.get_providers_registry")
def test_optimize_prompt_success(mock_get_registry, client):
    """Test /optimize-prompt endpoint success"""
    # Create a mock provider
    mock_provider = MagicMock(spec=ProviderBase)
    mock_provider.is_api_key_valid.return_value = True
    mock_provider.name = "google"
    mock_provider.key_env_var = "GOOGLE_API_KEY"
    mock_provider.generate_content.return_value = "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    
    # Create a mock registry
    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_registry.available_providers.return_value = [mock_provider]
    mock_get_registry.return_value = mock_registry
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_api_key"}):
        response = client.post("/api/optimize-prompt", json={"request": "test request", "provider": "google"})
        
        assert response.status_code == 200
        assert response.json == {
            "optimized_prompt": "<optimized_prompt>Test optimized prompt</optimized_prompt>"
        }
        mock_provider.generate_content.assert_called_once()


@patch("api.get_providers_registry")
def test_optimize_prompt_no_api_key(mock_get_registry, client):
    """Test /optimize-prompt endpoint when no API key is set"""
    # Create a mock provider
    mock_provider = MagicMock(spec=ProviderBase)
    mock_provider.is_api_key_valid.return_value = False
    mock_provider.name = "google"
    mock_provider.key_env_var = "GOOGLE_API_KEY"
    
    # Create a mock registry
    mock_registry = MagicMock()
    mock_registry.get_provider.return_value = mock_provider
    mock_registry.available_providers.return_value = [mock_provider]
    mock_get_registry.return_value = mock_registry
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
        response = client.post("/api/optimize-prompt", json={"request": "test request", "provider": "google"})
        
        assert response.status_code == 400
        assert "Provider API key not valid" in response.json["error"]


# Keep the existing test as it's already correct
def test_optimize_prompt_no_request(client):
    response = client.post("/api/optimize-prompt", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}


# Keep the existing test as it's already correct
def test_optimize_prompt_empty_request(client):
    response = client.post("/api/optimize-prompt", json={"request": ""})
    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}


def test_optimize_prompt_provider_error(client, mock_providers):
    """Test provider error handling"""
    # Unpack the mock provider and registry
    mock_provider, mock_registry = mock_providers
    mock_provider.generate_content.side_effect = Exception("API Error")
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
        response = client.post(
            "/api/optimize-prompt",
            json={"request": "Test error", "provider": "google"}
        )
        
        assert response.status_code == 500
        assert "API Error" in response.json["error"]


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
        "/api/pea/start", json={"initial_request": "Help me create a prompt", "provider": "google"}
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
    assert len(pea_conversations[session_id]["history"]) == 3


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

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me", "provider": "google"})
    session_id = start_response.json["session_id"]

    # Then test chat endpoint
    mock_response.text = "Follow-up response"
    chat_response = client.post(
        "/api/pea/chat",
        json={"session_id": session_id, "message": "My follow-up question", "provider": "google"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json == {"response": "Follow-up response"}
    # After the initial 3 messages, adding a user and model message makes it 5
    assert len(pea_conversations[session_id]["history"]) == 5


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

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me", "provider": "google"})
    session_id = start_response.json["session_id"]

    # Then test finalize endpoint
    mock_response.text = "<optimized_prompt>Final XML prompt</optimized_prompt>"
    finalize_response = client.post(
        "/api/pea/finalize", json={"session_id": session_id, "provider": "google"}
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
    response = client.post("/api/pea/start", json={"initial_request": "Help me", "provider": "google"})
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

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me", "provider": "google"})
    session_id = start_response.json["session_id"]

    # Then test chat endpoint with API error
    chat_response = client.post(
        "/api/pea/chat",
        json={"session_id": session_id, "message": "My follow-up question", "provider": "google"},
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

    start_response = client.post("/api/pea/start", json={"initial_request": "Help me", "provider": "google"})
    session_id = start_response.json["session_id"]
    client.post("/api/pea/chat", json={"session_id": session_id, "message": "Turn 1", "provider": "google"})

    # Then test finalize endpoint with API error
    finalize_response = client.post(
        "/api/pea/finalize", json={"session_id": session_id, "provider": "google"}
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