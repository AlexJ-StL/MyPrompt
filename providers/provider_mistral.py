from providers.provider_base import ProviderBase
import requests
import logging

class MistralProvider(ProviderBase):
    key_env_var = "MISTRAL_API_KEY"

    def _get_default_model(self) -> str:
        return "mistral-large-latest"

    def format_messages(self, chat_history: list) -> list:
        # Mistral API uses the same format as OpenAI
        return chat_history

    def _make_request(self, url: str, data: dict) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response

    def generate_content(self, chat_history: list) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        messages = self.format_messages(chat_history)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4000,
        }
        response = self._make_request(url, payload)
        return response.json()["choices"][0]["message"]["content"].strip()
