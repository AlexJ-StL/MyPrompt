import os
import logging

from providers.provider_base import ProviderBase


class ProviderRegistry:
    def __init__(self, *providers: ProviderBase):
        self._logger = logging.getLogger(__name__)
        self._providers = providers
        self._environment_vars = {
            provider.key_env_var: provider for provider in providers
        }

    def available_providers(self) -> list[ProviderBase]:
        return [provider for provider in self._providers if provider.is_api_key_valid()]

    def get_active_providers(self) -> list[ProviderBase]:
        return self.available_providers()

    def get_provider(self, provider_name: str) -> ProviderBase:
        for provider in self._providers:
            if provider.name == provider_name:
                return provider
        raise ValueError(f"Provider {provider_name} not found")
