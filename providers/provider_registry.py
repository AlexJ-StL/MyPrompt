import os
import logging

from providers.provider_base import ProviderBase


class ProviderRegistry:
    def __init__(self, *providers: ProviderBase):
        self._logger = logging.getLogger(__name__)
        self._providers = providers
        self._environment_vars = {
            provider.env_var_name: provider for provider in providers
        }

    def available_providers(self) -> list[ProviderBase]:
        return [provider for provider in self._providers if provider.is_api_key_valid()]

    def get_provider(self, api_key: str) -> ProviderBase:
        provider_key = os.environ.get(api_key)
        if provider_key not in self._environment_vars or not os.environ.get(
            provider_key
        ):
            logging.error(f"Invalid or missing API key: {api_key}")
            raise ValueError(f"Invalid or missing API key: {api_key}")

        return self._environment_vars[provider_key]
