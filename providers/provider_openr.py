from providers.provider_base import ProviderBase
from typing import List, Dict
import requests

class OpenRouterProvider(ProviderBase):
    key_env_var = "OPENROUTER_API_KEY"

    def _get_default_model(self) -> str:
        return "google/gemini-pro"

    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        # OpenRouter API uses the same format as OpenAI
        return chat_history

    def _make_request(self, url: str, data: Dict) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://myprompt.alexjekop.com",
            "X-Title": "MyPrompt Assistant"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, chat_history: List[Dict]) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        messages = self.format_messages(chat_history)
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        response = self._make_request(url, payload)
        return response.json()["choices"][0]["message"]["content"].strip()