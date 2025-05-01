import pytest
from unittest.mock import patch, MagicMock
from .api import api_bp
from flask import Flask

@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('myprompt.api.genai.GenerativeModel')
@patch('myprompt.api.os.getenv')
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
    mock_GenerativeModel.assert_called_once_with(
        model_name="gemini-2.5-flash-preview-04-17",
        safety_settings={
            0: 0, # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            1: 0, # HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
            2: 0, # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
            3: 0, # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    mock_model_instance.generate_content.assert_called_once()
    # Further assertion on the prompt content can be added if needed

@patch('myprompt.api.os.getenv')
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

@patch('myprompt.api.genai.GenerativeModel')
@patch('myprompt.api.os.getenv')
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