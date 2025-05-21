import pytest
from unittest.mock import patch, MagicMock
from api import api_bp, pea_conversations
from flask import Flask

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_optimize_prompt_success(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "<optimized_prompt>Test optimized prompt</optimized_prompt>"
    mock_model_instance.generate_content.return_value = mock_response

    response = client.post('/api/optimize-prompt', json={'request': 'test request'})

    assert response.status_code == 200
    assert response.json == {"optimized_prompt": "<optimized_prompt>Test optimized prompt</optimized_prompt>"}
    mock_getenv.assert_called_once_with('GEMINI_API_KEY')
    # Verify the model was created with correct settings
    mock_GenerativeModel.assert_called_once()
    call_args = mock_GenerativeModel.call_args
    assert call_args.kwargs['model_name'] == "gemini-2.5-flash-preview-04-17"
    assert isinstance(call_args.kwargs['safety_settings'], dict)
    assert len(call_args.kwargs['safety_settings']) == 4
    mock_model_instance.generate_content.assert_called_once()

@patch('api.os.getenv')
def test_optimize_prompt_no_api_key(mock_getenv, client):
    mock_getenv.return_value = None

    response = client.post('/api/optimize-prompt', json={'request': 'test request'})

    assert response.status_code == 500
    assert response.json == {"error": "GEMINI_API_KEY not set"}
    mock_getenv.assert_called_once_with('GEMINI_API_KEY')

def test_optimize_prompt_no_request(client):
    response = client.post('/api/optimize-prompt', json={})

    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}

def test_optimize_prompt_empty_request(client):
    response = client.post('/api/optimize-prompt', json={'request': ''})

    assert response.status_code == 400
    assert response.json == {"error": "No request provided"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_optimize_prompt_api_error(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_model_instance.generate_content.side_effect = Exception("API Error")

    response = client.post('/api/optimize-prompt', json={'request': 'test request'})

    assert response.status_code == 500
    assert response.json == {"error": "API Error"}
    mock_getenv.assert_called_once_with('GEMINI_API_KEY')
    mock_GenerativeModel.assert_called_once()
    mock_model_instance.generate_content.assert_called_once()

# PEA Tests
@pytest.fixture(autouse=True)
def clear_conversations():
    """Clear the conversation store before and after each test."""
    pea_conversations.clear()
    yield
    pea_conversations.clear()

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_start_pea_session_success(mock_getenv, mock_GenerativeModel, client):
    # Setup mocks
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Hello! I'm PEA. Let me help you refine your prompt."
    mock_model_instance.generate_content.return_value = mock_response

    # Test request
    response = client.post('/api/pea/start', json={'initial_request': 'Help me create a prompt'})

    # Assertions
    assert response.status_code == 200
    assert 'session_id' in response.json
    assert 'response' in response.json
    assert response.json['response'] == mock_response.text
    
    # Verify conversation was stored
    session_id = response.json['session_id']
    assert session_id in pea_conversations
    # After the initial turn, history should contain 2 messages: user (with system prompt prepended) + model response
    assert len(pea_conversations[session_id]) == 2

def test_start_pea_session_no_request(client):
    response = client.post('/api/pea/start', json={})
    assert response.status_code == 400
    assert response.json == {"error": "No initial request provided"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_chat_success(mock_getenv, mock_GenerativeModel, client):
    # First start a session
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Initial response"
    mock_model_instance.generate_content.return_value = mock_response

    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']

    # Then test chat endpoint
    mock_response.text = "Follow-up response"
    chat_response = client.post('/api/pea/chat', json={
        'session_id': session_id,
        'message': 'My follow-up question'
    })

    assert chat_response.status_code == 200
    assert chat_response.json == {"response": "Follow-up response"}
    # After the initial 2 messages, adding a user and model message makes it 4
    assert len(pea_conversations[session_id]) == 4

def test_pea_chat_invalid_session(client):
    response = client.post('/api/pea/chat', json={
        'session_id': 'invalid_session_id',
        'message': 'test message'
    })
    assert response.status_code == 400
    assert response.json == {"error": "Invalid session ID"}

def test_pea_chat_missing_data(client):
    response = client.post('/api/pea/chat', json={})
    assert response.status_code == 400
    assert response.json == {"error": "Missing session_id or message"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_finalize_prompt_success(mock_getenv, mock_GenerativeModel, client):
    # First start a session
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response = MagicMock()
    mock_response.text = "Initial response"
    mock_model_instance.generate_content.return_value = mock_response

    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']

    # Then test finalize endpoint
    mock_response.text = "<optimized_prompt>Final XML prompt</optimized_prompt>"
    finalize_response = client.post('/api/pea/finalize', json={'session_id': session_id})

    assert finalize_response.status_code == 200
    assert finalize_response.json == {"final_prompt": "<optimized_prompt>Final XML prompt</optimized_prompt>"}
    assert session_id not in pea_conversations  # Session should be cleared after finalization

def test_finalize_prompt_invalid_session(client):
    response = client.post('/api/pea/finalize', json={'session_id': 'invalid_session_id'})
    assert response.status_code == 400
    assert response.json == {"error": "Invalid session ID"}

def test_finalize_prompt_missing_session_id(client):
    response = client.post('/api/pea/finalize', json={})
    assert response.status_code == 400
    assert response.json == {"error": "No session_id provided"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_api_error(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_model_instance.generate_content.side_effect = Exception("API Error")

    # Test API error during PEA start
    response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    assert response.status_code == 500
    assert response.json == {"error": "API Error"}
    
    # Test API error during PEA chat (need a valid session first)
    # Start a session successfully first
    mock_response_start = MagicMock()
    mock_response_start.text = "Initial PEA response"
    mock_model_instance.generate_content.side_effect = [mock_response_start, Exception("API Error")] # First call succeeds, second fails
    
    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']

    # Then test chat endpoint with API error
    chat_response = client.post('/api/pea/chat', json={
        'session_id': session_id,
        'message': 'My follow-up question'
    })

    assert chat_response.status_code == 500
    assert chat_response.json == {"error": "API Error"}
    
    # Test API error during PEA finalize (need a valid session and some history)
    # Start a session and add a chat turn successfully first
    mock_response_chat = MagicMock()
    mock_response_chat.text = "Chat response"
    mock_model_instance.generate_content.side_effect = [mock_response_start, mock_response_chat, Exception("API Error")] # First two succeed, third fails
    
    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']
    client.post('/api/pea/chat', json={'session_id': session_id, 'message': 'Turn 1'})

    # Then test finalize endpoint with API error
    finalize_response = client.post('/api/pea/finalize', json={'session_id': session_id})

    assert finalize_response.status_code == 500
    assert finalize_response.json == {"error": "API Error"}
    # Session should NOT be cleared if finalize fails due to API error
    assert session_id in pea_conversations


def test_handle_bad_request(client):
    # Test the error handler for BadRequest (e.g., invalid JSON)
    # Sending non-JSON data to a JSON-only endpoint should trigger this
    response = client.post('/api/optimize-prompt', data='not json')
    # Expecting a 415 because the UnsupportedMediaType handler will be triggered first
    assert response.status_code == 415
    # The response body will be handled by the specific error handler now
    assert response.json == {"error": "Unsupported Media Type: Content-Type must be application/json"} # Based on the error handler logic

def test_handle_unsupported_media_type(client):
    # Test the error handler for UnsupportedMediaType
    # Sending data with a wrong Content-Type should trigger this
    response = client.post('/api/optimize-prompt', data={'request': 'test'}, headers={'Content-Type': 'text/plain'})
    assert response.status_code == 415
    assert response.json == {"error": "Unsupported Media Type: Content-Type must be application/json"}

# Add more detailed PEA interaction tests
@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_multi_turn_interaction(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    
    # Mock responses for multiple turns
    mock_responses = [
        MagicMock(text="Okay, what is your main goal?"), # First response
        MagicMock(text="Got it. And who is the target audience?"), # Second response
        MagicMock(text="Thanks. Can you provide details about the offer?"), # Third response
        MagicMock(text="<optimized_prompt>Final Prompt</optimized_prompt>") # Final response
    ]
    mock_model_instance.generate_content.side_effect = mock_responses

    # Start session
    start_response = client.post('/api/pea/start', json={'initial_request': 'Help with email prompt'})
    session_id = start_response.json['session_id']
    assert start_response.json['response'] == mock_responses[0].text
    assert len(pea_conversations[session_id]) == 2 # user (with system) + model

    # First chat turn
    chat_response1 = client.post('/api/pea/chat', json={'session_id': session_id, 'message': 'My goal is to drive sales.'})
    assert chat_response1.status_code == 200
    assert chat_response1.json['response'] == mock_responses[1].text
    assert len(pea_conversations[session_id]) == 4 # + user + model
    
    # Second chat turn
    chat_response2 = client.post('/api/pea/chat', json={'session_id': session_id, 'message': 'Target audience is existing customers.'})
    assert chat_response2.status_code == 200
    assert chat_response2.json['response'] == mock_responses[2].text
    assert len(pea_conversations[session_id]) == 6 # + user + model

    # Finalize prompt
    finalize_response = client.post('/api/pea/finalize', json={'session_id': session_id})
    assert finalize_response.status_code == 200
    assert finalize_response.json['final_prompt'] == mock_responses[3].text
    assert session_id not in pea_conversations # Session cleared

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_start_with_api_key_missing(mock_getenv, mock_GenerativeModel, client):
    mock_getenv.return_value = None
    response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    assert response.status_code == 500
    assert response.json == {"error": "GEMINI_API_KEY not set"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_chat_with_api_key_missing(mock_getenv, mock_GenerativeModel, client):
    # Start a session successfully first
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response_start = MagicMock()
    mock_response_start.text = "Initial PEA response"
    mock_model_instance.generate_content.return_value = mock_response_start
    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']
    
    # Then test chat endpoint with missing API key (mock getenv after session start)
    mock_getenv.return_value = None
    chat_response = client.post('/api/pea/chat', json={
        'session_id': session_id,
        'message': 'My follow-up question'
    })
    assert chat_response.status_code == 500
    assert chat_response.json == {"error": "GEMINI_API_KEY not set"}

@patch('api.genai.GenerativeModel')
@patch('api.os.getenv')
def test_pea_finalize_with_api_key_missing(mock_getenv, mock_GenerativeModel, client):
     # Start a session successfully first
    mock_getenv.return_value = 'fake_api_key'
    mock_model_instance = MagicMock()
    mock_GenerativeModel.return_value = mock_model_instance
    mock_response_start = MagicMock()
    mock_response_start.text = "Initial PEA response"
    mock_model_instance.generate_content.return_value = mock_response_start
    start_response = client.post('/api/pea/start', json={'initial_request': 'Help me'})
    session_id = start_response.json['session_id']
    
    # Then test finalize endpoint with missing API key (mock getenv after session start)
    mock_getenv.return_value = None
    finalize_response = client.post('/api/pea/finalize', json={'session_id': session_id})
    assert finalize_response.status_code == 500
    assert finalize_response.json == {"error": "GEMINI_API_KEY not set"}