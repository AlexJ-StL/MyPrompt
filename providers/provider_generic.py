"""
Defines the Generic Provider class for OpenAI-compatible APIs
"""

from typing import List, Dict, Optional
import requests
from providers.provider_base import ProviderBase


class GenericProvider(ProviderBase):
    """Handles generic providers with unified API structure"""

    def __init__(self, model_name: Optional[str]):
        super().__init__(model_name)

        # Base API URL should be set by specific implementations
        self.base_url = ""

    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        """
        Converts general message structure to provider-specific format
        Assumes first message is the user's request with content
        """
        formatted_messages = []
        for msg in chat_history:
            role = msg["role"]
            content = None

            # Handle cases with both 'content' and 'parts'
            if "content" in msg and msg["content"]:
                # Use the single content field
                content = str(msg["content"])
            elif "parts" in msg and msg["parts"] and "text" in msg["parts"][0]:
                # Assume only 1st text part contains the message
                content = str(msg["parts"][0]["text"])

            if not content:
                # Skip if message is empty but present
                continue

            # Append formatted message
            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def _make_request(self, url: str, data: Dict) -> requests.Response:
        """Sends API request to provider with Authorization header attached"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Check if provider has non-standard auth type (Anthropic)
            if self.key_env_var == "ANTHROPIC_API_KEY":
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        # Handle non-Bearer providers
        for provider, header in {
            "openrouter": "HTTP-Referer",
            "openrouter_title": "X-Title",
        }.items():
            env_var = f"{provider} env var here"
            if value := os.getenv(env_var):
                # Add to headers but prevent adding empty values
                if header_value and len(str(header_value) > 0):
                    headers[header] = str(header_value)

        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response


class OpenRouterProvider(GenericProvider):
    """Special handling for OpenRouter features"""

    key_env_var = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _make_request(self, url: str, data: Dict) -> requests.Response:
        # Add specific headers here?
        response = super()._make_request(url, data)
        # Check if response needs any special handling
        return response


class MistralProvider(GenericProvider):
    """Mistral-specific configurations"""

    key_env_var = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1/chat/completions"

    def format_messages(self, chat_history):
        """Mistral handles first item differently?"""
        # Same implementation as Generic for now
        return super().format_messages(chat_history)
