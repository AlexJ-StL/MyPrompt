from providers.provider_base import ProviderBase
import requests
import logging
from typing import Optional

class LMStudioProvider(ProviderBase):
    key_env_var = None  # No API key required

    def _get_default_model(self) -> str:
        return "llama2"
        
    def _get_api_key(self) -> Optional[str]:
        """LM Studio doesn't require an API key"""
        return None

    def format_messages(self, chat_history: list) -> list:
        # LMStudio uses the same format as OpenAI
        return chat_history

    def _make_request(self, url: str, data: dict) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, chat_history: list) -> str:
        url = "http://localhost:1234/v1/chat/completions"
        messages = self.format_messages(chat_history)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4000,
        }
        response = self._make_request(url, payload)
        return response.json()["choices"][0]["message"]["content"].strip()
