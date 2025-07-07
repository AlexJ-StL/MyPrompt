from flask import Flask, jsonify, g
from api import api_bp
from provider_registry import ProviderRegistry
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix="/api")

PROVIDERS = [
    SomeProviderClass("openai"),
    SomeProviderClass("anthropic"),
    SomeProviderClass("google"),
    SomeProviderClass("openrouter"),
]
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
