"""App Factory to initialize providers using environment variables"""

import os
from typing import Type, Dict, Any, Awaitable, Callable
from flask import Flask, g
from providers.provider_base import ProviderBase
from providers.provider_registry import ProviderRegistry
# Import specific providers - they will be initialized in the registry
from providers.provider_openai import OpenAiProvider as OpenAIProvider
from providers.provider_anthropic import AnthropicProvider
from providers.provider_google import GoogleProvider
from providers.provider_openr import OpenRouterProvider
from providers.provider_groq import GroqProvider
from providers.provider_mistral import MistralProvider
from providers.provider_ollama import OllamaProvider
from providers.provider_lmstudio import LMStudioProvider

def create_app() -> Flask:
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    
    # Initialize provider registry with all provider types
    providers = [
        OpenAIProvider(model_name="gpt-4"),
        AnthropicProvider(model_name="claude-3-opus-20240229"),
        GoogleProvider(model_name="gemini-pro"),
        OpenRouterProvider(model_name="google/gemini-pro"),
        GroqProvider(model_name="llama3-8b-8192"),
        MistralProvider(model_name="mistral-large-latest"),
        OllamaProvider(model_name="llama2"),
        LMStudioProvider(model_name="llama2"),
    ]
    
    provider_registry = ProviderRegistry(*providers)
    
    # Add provider registry to app context
    @app.before_request
    def before_request():
        # Make provider registry available in request context
        g.providers = provider_registry
        # Debug print to verify this is being called
        logging.debug("before_request: Set g.providers")
    
    # Import and register blueprints
    from api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "healthy"}, 200
        
    return app
