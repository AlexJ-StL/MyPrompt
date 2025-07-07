"""
Main application initialization and routing

This file:
  - Initializes the Flask app
  - Registers all API endpoints
  - Configures the provider registry
"""

import os
from flask import Flask, jsonify, g
from dotenv import load_dotenv
from api import api_bp
from app_factory import get_active_providers
from provider_base import ProviderBase
from provider_generic import GenericProvider
from provider_anthropic import AnthropicProvider
from provider_openai import OpenAiProvider

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix="/api")


# Pre-initialize providers to handle env var checks and validations
@app.before_request
def load_providers():
    """
    Middleware to pre-load available providers based on environment variables
    Providers should validate their own API keys and credentials
    """
    global ACTIVE_PROVIDERS
    ACTIVE_PROVIDERS = get_active_providers()
    g.providers = {p.name().lower(): p for p in ACTIVE_PROVIDERS}


# Helper to normalize request data across providers
def extract_pea_request_data(request):
    """Common request parser for /pea/ endpoints"""
    try:
        data = request.json
        req_id = data.get("session_id")
        user_message = data.get("message")
        provider_key = data.get("provider", "google").lower()
        return req_id, user_message, provider_key
    except:
        return None, None, None


# Error handler to catch KeyErrors during provider lookup
@app.errorhandler(KeyError)
def handle_key_error(error):
    missing_param = error.args[0]
    return jsonify({"error": f"Invalid request - missing {missing_param}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=3005)
