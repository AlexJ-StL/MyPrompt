from flask import Flask, jsonify, g
from api import api_bp
from providers.provider_registry import ProviderRegistry
from providers.provider_openai import OpenAiProvider  # Corrected capitalization
from providers.provider_anthropic import AnthropicProvider
from providers.provider_google import GoogleProvider
from providers.provider_openr import (
    OpenRouterProvider,
)  # Match exact class name capitalization
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix="/api")

PROVIDERS = [
    OpenAiProvider(model_name="gpt-4"),  # Match exact class name capitalization
    AnthropicProvider(model_name="claude-3-opus"),
    GoogleProvider(model_name="gemini-pro"),
    OpenRouterProvider(model_name="google/gemini-pro"),
]

# Remove any remaining references to SomeProviderClass
provider_registry = ProviderRegistry(*PROVIDERS)


@app.before_request
def load_providers():
    active_providers = provider_registry.get_active_providers()
    providers = provider_registry.available_providers()
    g.providers = providers


@app.route("/")
def ping():
    return jsonify({"ping": "pong"}), 200


@app.route("/providers")
def available_providers():
    providers = [p.name for p in g.providers]
    return jsonify(providers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=5001)
