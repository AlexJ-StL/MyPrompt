"""OpenAI Provider Interface Implementation"""

import requests
from providers.provider_base import ProviderBase


class OpenAiProvider(ProviderBase):
    """Interface adapter for the official OpenAI API

    - Handles API key management using env vars
    - Normalizes message history to {role, content} format
    - Routes endpoints through the unified provider layer
    """

    # API key/endpoint constants
    key_env_var = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1/chat/completions"
    max_tokens = 4000
    model_base = "o4-mini"  # Default can be overriden by config

    def _get_default_model(self) -> str:
        """Returns the default model, can be set via env"""
        if model := os.getenv("OPENAI_MODEL", self.model_base):
            return model
        return self.model_base

    def format_messages(self, chat_history: list[dict]) -> list[dict]:
        """Extracts the content from 'parts' field if present"""
        processed = []
        for msg in chat_history:
            role = msg["role"]
            if "parts" in msg:
                # Assume first part contains the text
                content = msg["parts"][0]["text"]
            else:
                content = msg["content"]
            processed.append({"role": role, "content": content})
        return processed

    def _make_request(self, data: dict) -> requests.Response:
        """Executes the API request with proper authorization"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, history: list[dict]) -> str:
        """Generates a response using the provider-formatted history"""
        formatted = self.format_messages(history)
        payload = {
            "model": self.model,
            "messages": formatted,
        }
        response_data = self._make_request(payload)
        # Extract first choice (assumed to be first valid element)
        choices = response_data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content")
            return content if isinstance(content, str) else "Error in response"
        else:
            return "Failed to get any completions"
