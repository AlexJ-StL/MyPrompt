"""OpenAI Provider Interface Implementation"""

from .provider_base import ProviderBase


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
    model_base = "gpt-4"  # Default can be overriden by config

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

    def _make_request(self, data: dict) -> dict:
        """Executes the API request with proper authorization"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = self.base_url.format(model=model)
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def generate_response(self, history: list[dict]) -> str:
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
