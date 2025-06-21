import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest
from flask import Flask

# Import the Flask app from the app package
from app import app as flask_app


@pytest.fixture
def client():
    """
    Pytest fixture to provide a test client for the Flask app.
    """
    with flask_app.test_client() as client:
        yield client


class TestIndex:
    """
    Test suite for the 'index' route in app.py.
    """

    @pytest.mark.happy_path
    def test_index_route_returns_expected_string(self, client):
        """
        Happy Path: Ensure GET / returns the expected string response.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert response.data == b"MyPrompt Backend"
        assert response.mimetype == "text/html"

    @pytest.mark.happy_path
    def test_index_route_allows_cors(self, client):
        """
        Happy Path: Ensure CORS headers are present in the response.
        """
        response = client.get("/", headers={"Origin": "http://example.com"})
        # CORS should allow all origins
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    @pytest.mark.happy_path
    def test_index_returns_welcome_message(self, client):
        """
        Test that the root endpoint ('/') returns the expected welcome message.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert response.data.decode("utf-8") == "MyPrompt Backend"

    @pytest.mark.happy_path
    def test_index_response_content_type(self, client):
        """
        Test that the root endpoint returns a response with 'text/html' content type.
        """
        response = client.get("/")
        assert response.content_type.startswith("text/html")

    @pytest.mark.edge_case
    def test_index_route_method_not_allowed(self, client):
        """
        Edge Case: Ensure non-GET methods (e.g., POST) are not allowed on the index route.
        """
        response = client.post("/")
        assert response.status_code == 405  # Method Not Allowed

    @pytest.mark.edge_case
    @pytest.mark.edge_case
    def test_index_route_not_found(self, client):
        """
        Edge Case: Ensure a non-existent route returns 404.
        """
        response = client.get("/nonexistent")
        assert response.status_code == 404

    @pytest.mark.happy_path
    def test_index_route_case_sensitivity(self, client):
        """
        Happy Path: Ensure route is case-sensitive and /INDEX does not match /.
        """
        response = client.get("/INDEX")
        assert response.status_code == 404

    @pytest.mark.edge_case
    def test_index_route_head_request(self, client):
        """
        Edge Case: Ensure HEAD requests to / return correct status and no body.
        """
        response = client.head("/")
        assert response.status_code == 200
        # HEAD responses have no body
        assert not response.data

    @pytest.mark.edge_case
    def test_index_with_query_parameters(self, client):
        """
        Test that the root endpoint ignores query parameters and still returns the welcome message.
        """
        response = client.get("/?foo=bar&baz=qux")
        assert response.status_code == 200
        assert response.data.decode("utf-8") == "MyPrompt Backend"

    @pytest.mark.edge_case
    def test_index_with_post_method(self, client):
        """
        Test that the root endpoint does not allow POST requests and returns 405 Method Not Allowed.
        """
        response = client.post("/")
        assert response.status_code == 405

    @pytest.mark.edge_case
    def test_index_with_unsupported_method(self, client):
        """
        Test that the root endpoint does not allow PUT requests and returns 405 Method Not Allowed.
        """
        response = client.put("/")
        assert response.status_code == 405

    @pytest.mark.edge_case
    def test_index_with_headers(self, client):
        """
        Test that the root endpoint returns the same response even when custom headers are sent.
        """
        response = client.get("/", headers={"X-Test-Header": "test-value"})
        assert response.status_code == 200
        assert response.data.decode("utf-8") == "MyPrompt Backend"
