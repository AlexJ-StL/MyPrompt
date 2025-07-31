"""App Factory to initialize providers using environment variables"""

import os
import logging
from typing import Type, Dict, Any, Awaitable, Callable, Optional, List
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

def create_app(
    provider_registry_instance: Optional[ProviderRegistry] = None,
    provider_list: Optional[List[ProviderBase]] = None
) -> Flask:
    """
    Factory function to create and configure the Flask app.
    
    Args:
        provider_registry_instance: Optional ProviderRegistry instance to use.
            If provided, this instance will be used instead of creating a new one.
        provider_list: Optional list of ProviderBase instances to register.
            If provided, these providers will be registered in the registry.
            Ignored if provider_registry_instance is provided.
            
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Use provided registry or create new one
    if provider_registry_instance is not None:
        provider_registry = provider_registry_instance
        logging.info("Using provided provider registry instance")
    else:
        # Create list of providers if not provided
        if provider_list is None:
            provider_list = [
                OpenAIProvider(model_name="gpt-4"),
                AnthropicProvider(model_name="claude-3-opus-20240229"),
                GoogleProvider(model_name="gemini-1.5-flash"),
                OpenRouterProvider(model_name="google/gemini-pro"),
                GroqProvider(model_name="llama3-8b-8192"),
                MistralProvider(model_name="mistral-large-latest"),
                OllamaProvider(model_name="llama2"),
                LMStudioProvider(model_name="llama2"),
            ]
        
        provider_registry = ProviderRegistry(*provider_list)
        logging.info(f"Created provider registry with {len(provider_list)} providers")
    
    # Import and register blueprints
    from api import api_bp, set_provider_registry
    set_provider_registry(provider_registry)
    
    # Add provider registry to app context
    @app.before_request
    def before_request():
        # Make provider registry available in request context
        g.providers = provider_registry
        # Debug print to verify this is being called
        app.logger.debug("before_request: Set g.providers")
    
    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "healthy"}, 200
        
    app.register_blueprint(api_bp, url_prefix="/api")
    
    return app
