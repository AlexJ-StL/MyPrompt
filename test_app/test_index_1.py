# test_app_index.py

import pytest
from flask import Flask
from app import app as flask_app

@pytest.fixture
def client():
    """
    Pytest fixture to create a test client for the Flask app.
    """
    with flask_app.test_client() as client:
        yield client

class TestIndex:
    @pytest.mark.happy_path
    def test_index_returns_welcome_message(self, client):
        """
        Test that the root endpoint ('/') returns the expected welcome message.
        """
        response = client.get('/')
        assert response.status_code == 200
        assert response.data.decode('utf-8') == "MyPrompt Backend"

    @pytest.mark.happy_path
    def test_index_response_content_type(self, client):
        """
        Test that the root endpoint returns a response with 'text/html' content type.
        """
        response = client.get('/')
        assert response.content_type.startswith('text/html')

    @pytest.mark.edge_case
    def test_index_with_trailing_slash(self, client):
        """
        Test that accessing the root endpoint with a trailing slash returns the same response.
        """
        response = client.get('//')
        # Flask normalizes '//' to '/'
        assert response.status_code == 200
        assert response.data.decode('utf-8') == "MyPrompt Backend"

    @pytest.mark.edge_case
    def test_index_with_query_parameters(self, client):
        """
        Test that the root endpoint ignores query parameters and still returns the welcome message.
        """
        response = client.get('/?foo=bar&baz=qux')
        assert response.status_code == 200
        assert response.data.decode('utf-8') == "MyPrompt Backend"

    @pytest.mark.edge_case
    def test_index_with_post_method(self, client):
        """
        Test that the root endpoint does not allow POST requests and returns 405 Method Not Allowed.
        """
        response = client.post('/')
        assert response.status_code == 405

    @pytest.mark.edge_case
    def test_index_with_unsupported_method(self, client):
        """
        Test that the root endpoint does not allow PUT requests and returns 405 Method Not Allowed.
        """
        response = client.put('/')
        assert response.status_code == 405

    @pytest.mark.edge_case
    def test_index_with_headers(self, client):
        """
        Test that the root endpoint returns the same response even when custom headers are sent.
        """
        response = client.get('/', headers={'X-Test-Header': 'test-value'})
        assert response.status_code == 200
        assert response.data.decode('utf-8') == "MyPrompt Backend"