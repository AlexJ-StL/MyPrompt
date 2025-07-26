from providers.provider_base import ProviderBase
from typing import List, Dict
import requests

class AnthropicProvider(ProviderBase):
    key_env_var = "ANTHROPIC_API_KEY"

    def _get_default_model(self) -> str:
        return "claude-3-opus-20240229"

    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        # Anthropic API uses a slightly different format for messages
        return chat_history

    def _make_request(self, url: str, data: Dict) -> requests.Response:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, chat_history: List[Dict]) -> str:
        url = "https://api.anthropic.com/v1/messages"
        messages = self.format_messages(chat_history)
        payload = {
            "model": self.model_name,
            "max_tokens": 4000,
            "messages": messages,
        }
        response = self._make_request(url, payload)
        return response.json()["content"][0]["text"].strip()