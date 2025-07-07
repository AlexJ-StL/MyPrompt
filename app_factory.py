"""App Factory to initialize providers using environment variables"""

from typing import Type, Dict
from .providers.provider_base import ProviderBase
from .providers.provider_generic import GenericProvider
from .providers.provider_anthropic import AnthropicProvider

# Register providers here, keyed by environment variable keys
PROVIDERS: Dict[str, Type[ProviderBase]] = {
    "ANTHROPIC_API_KEY": AnthropicProvider,
    "OPENROUTER_API_KEY": GenericProvider,
    "GROQ_API_KEY": GenericProvider,
    "MISTRAL_API_KEY": GenericProvider,
    # Additional providers would be added here with their env var key
}


def get_active_providers() -> Dict[Type[ProviderBase], ProviderBase]:
    """
    Scans environment variables and initializes all available providers
    whose API keys are present in the environment.
    """
    active_providers = {}
    for env_key, provider_class in PROVIDERS.items():
        if api_key := os.getenv(env_key):
            # Initialize provider instance with default model
            # Can override via environment vars in the future if needed
            provider_instance = provider_class()
            active_providers[provider_class] = provider_instance
    return active_providers
