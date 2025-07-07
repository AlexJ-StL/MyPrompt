from providers.provider_base import ProviderBase


class AnthropicProvider(ProviderBase):
    API_KEY_ENV = "ANTHROPIC_API_KEY"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def call(
        self, message: str, temperature: float = 0.4, max_tokens: int = 150
    ) -> dict:
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": "claude-4-sonnet",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": message},
            ],
        }
        response = self.send_request(url, headers, payload)
        return response
