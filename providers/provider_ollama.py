from providers.provider_base import ProviderBase
import requests
import logging
from typing import Optional

class OllamaProvider(ProviderBase):
    key_env_var = None  # No API key required

    def _get_default_model(self) -> str:
        return "llama2"

    def _get_api_key(self) -> Optional[str]:
        """Ollama doesn't require an API key"""
        return None
        
    def format_messages(self, chat_history: list) -> list:
        # For simple cases, just use the last message content
        # In a real implementation, you might want to handle the full history
        if chat_history:
            return [{"role": "user", "content": chat_history[-1]["content"]}]
        return []

    def _make_request(self, url: str, data: dict) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, chat_history: list) -> str:
        url = "http://localhost:11434/api/generate"
        messages = self.format_messages(chat_history)
        payload = {
            "model": self.model_name,
            "prompt": messages[0]["content"] if messages else "",
            "stream": False
        }
        response = self._make_request(url, payload)
        return response.json()["response"].strip()
