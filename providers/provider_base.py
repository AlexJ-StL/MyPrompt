"""
This module defines the base class for interacting with various LLM providers.
Concrete implementations must be created to handle specific provider APIs.
"""

import os
import abc
import logging
from typing import List, Dict, Optional
import requests

# Configuration for logging
logging.basicConfig(level=logging.DEBUG)


class ProviderBase(abc.ABC):
    """
    Base class for LLM provider interface.

    Concrete implementations must:
    - Define their supported models
    - Implement message formatting
    - Override _make_request() for specific API calls
    """

    def __init__(self, model_name: Optional[str]):
        """
        Initializes provider with API key and model selection
        """
        self.model_name = model_name
        self.api_key = self._get_api_key()
        self.base_url = ""
        # Models are either a dict of models or a string
        # e.g. "mistral" or ["all" models]
        self.models = []
        self.name = self.__class__.__name__.lower().replace("provider", "")

        # If no model_name is provided, get default from class property
        if not model_name:
            self.model_name = self._get_default_model()

    @property
    @abc.abstractmethod
    def key_env_var(self) -> str:
        """Env var name to retrieve API key from"""
        raise NotImplementedError(
            "ProviderBase subclasses must define required key environment variable"
        )

    @property
    def fallback_env_var(self) -> Optional[str]:
        """
        Optional fallback variable used when primary key isn't set.
        Returns None to disable fallback.
        """
        return None  # Default no-fallback

    @abc.abstractmethod
    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        """Converts general message structure to provider-specific format"""
        raise NotImplementedError("Providers must implement message formatting logic")

    def _get_api_key(self) -> Optional[str]:
        """
        Fetches API key from environment, using fallback if available
        """
        api_key = os.getenv(self.key_env_var)
        if not api_key and self.fallback_env_var:
            api_key = os.getenv(self.fallback_env_var)
        if not api_key:
            logging.error(f"API key not set for provider using {self.key_env_var}")
            return None
        return api_key

    def is_api_key_valid(self) -> bool:
        """Checks if the API key is set."""
        return self.api_key is not None

    @classmethod
    @abc.abstractmethod
    def _get_default_model(cls) -> str:
        """Returns the provider's default model name"""
        raise NotImplementedError("ProviderBase subclasses must define default model")

    @abc.abstractmethod
    def _make_request(self, url: str, data: Dict) -> requests.Response:
        """Handles making the actual POST request to the provider's API"""
        raise NotImplementedError("Concrete providers must implement request logic")

    @abc.abstractmethod
    async def generate_content(self, chat_history: List[Dict]) -> str:
        """
        Main interface for content generation, implementing standard error handling.
        Converts history, formats messages, makes request, parses response, and handles errors.
        """
        raise NotImplementedError(
            "Concrete providers must implement the full API workflow"
        )