import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Flask

# Import the blueprint and function to test
from api import api_bp


@pytest.fixture
def client():
    """
    Fixture to create a Flask test client with the api blueprint registered.
    """
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestOptimizePrompt:
    """
    Unit tests for the optimize_prompt API endpoint.
    """

    # ------------------- Happy Path Tests -------------------

    @pytest.mark.happy_path
    def test_optimize_prompt_success(self, client):
        """
        Test that a valid request with a proper 'request' field returns a 200 and the optimized prompt.
        """
        test_api_key = "dummy_key"
        test_user_request = "Write a poem about the sea."
        test_llm_response = (
            "<optimized_prompt><poem>Ode to the Sea...</poem></optimized_prompt>"
        )

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": test_api_key}),
            patch("api.genai.configure") as mock_configure,
            patch("api.genai.GenerativeModel") as mock_model_class,
        ):

            mock_model = MagicMock()
            mock_model.generate_content.return_value.text = test_llm_response
            mock_model_class.return_value = mock_model

            response = client.post(
                "/api/optimize-prompt",
                data=json.dumps({"request": test_user_request}),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert "optimized_prompt" in data
            assert data["optimized_prompt"] == test_llm_response
            mock_configure.assert_called_once_with(api_key=test_api_key)
            mock_model_class.assert_called_once()

    @pytest.mark.happy_path
    def test_optimize_prompt_success_with_whitespace_response(self, client):
        """
        Test that the endpoint strips whitespace from the LLM response.
        """
        test_api_key = "dummy_key"
        test_user_request = "Summarize the history of AI."
        test_llm_response = "   <optimized_prompt>AI history...</optimized_prompt>   "

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": test_api_key}),
            patch("api.genai.configure"),
            patch("api.genai.GenerativeModel") as mock_model_class,
        ):

            mock_model = MagicMock()
            mock_model.generate_content.return_value.text = test_llm_response
            mock_model_class.return_value = mock_model

            response = client.post(
                "/optimize-prompt",
                data=json.dumps({"request": test_user_request}),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["optimized_prompt"] == test_llm_response.strip()

    # ------------------- Edge Case Tests -------------------

    @pytest.mark.edge_case
    def test_optimize_prompt_missing_json(self, client):
        """
        Test that a request with content_type json but no body returns a 400 error.
        """
        response = client.post(
            "/api/optimize-prompt",
            # Omit data parameter to simulate no request body
            # data="", # Sending empty string can cause parsing issues
            content_type="application/json",
        )
        assert response.status_code == 400
        assert (
            response.content_type == "application/json"
        )  # Ensure the error response is JSON
        data = response.get_json()
        assert data is not None  # Ensure get_json() didn't return None
        assert data["error"] == "No request provided"

    @pytest.mark.edge_case
    def test_optimize_prompt_missing_request_field(self, client):
        """
        Test that a request with JSON but missing 'request' field returns a 400 error.
        """
        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"foo": "bar"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["error"] == "No request provided"

    @pytest.mark.edge_case
    def test_optimize_prompt_empty_request_field(self, client):
        """
        Test that a request with an empty 'request' field returns a 400 error.
        """
        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": ""}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["error"] == "No request provided"

    @pytest.mark.edge_case
    def test_optimize_prompt_missing_api_key(self, client):
        """
        Test that if GEMINI_API_KEY is not set, a 500 error is returned.
        """
        with patch.dict(os.environ, {}, clear=True):
            response = client.post(
                "/api/optimize-prompt",
                data=json.dumps({"request": "Test prompt"}),
                content_type="application/json",
            )
            assert response.status_code == 500
            data = response.get_json()
            assert (
                data["error"]
                == "GOOGLE_API_KEY or OPENAI_API_KEY not set in environment"
            )

    @pytest.mark.edge_case
    def test_optimize_prompt_genai_raises_exception(self, client):
        """
        Test that if the LLM call raises an exception, a 500 error is returned with the exception message.
        """
        test_api_key = "dummy_key"
        test_user_request = "Write a story."
        error_message = "LLM API failure"

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": test_api_key}),
            patch("api.genai.configure"),
            patch("api.genai.GenerativeModel") as mock_model_class,
        ):

            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception(error_message)
            mock_model_class.return_value = mock_model

            response = client.post(
                "/api/optimize-prompt",
                data=json.dumps({"request": test_user_request}),
                content_type="application/json",
            )

            assert response.status_code == 500
            data = response.get_json()
            assert data["error"] == error_message

    @pytest.mark.edge_case
    def test_optimize_prompt_non_json_content_type(self, client):
        """
        Test that a request with a non-JSON content type returns a 415 error.
        """
        response = client.post(
            "/api/optimize-prompt",
            data="request=hello",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 415  # Expect Unsupported Media Type
        assert (
            response.content_type == "application/json"
        )  # Ensure the error response is JSON
        data = response.get_json()
        assert data is not None
        assert (
            data["error"]
            == "Unsupported Media Type: Content-Type must be application/json"
        )  # Or a similar appropriate message

    @pytest.mark.edge_case
    def test_optimize_prompt_request_field_is_none(self, client):
        """
        Test that a request with 'request' field explicitly set to None returns a 400 error.
        """
        response = client.post(
            "/api/optimize-prompt",
            data=json.dumps({"request": None}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["error"] == "No request provided"
