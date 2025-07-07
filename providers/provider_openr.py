from providers.provider_base import ProviderBase


class OpenRouterProvider(ProviderBase):
    API_KEY_ENV = "OPENROUTER_API_KEY"

    def __init__(self, api_key) -> None:
        if not api_key:
            raise ValueError("The API key is required to create a OpenRouter provider.")
        self.api_key = api_key

    def call(self, message):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        data = {
            "model": "some-model",
            "messages": [{"role": "user", "content": message.content}],
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except Exception as e:
            logging.error(f"OpenRouter provider call failed: {e}")
            return {"Error": str(e)}
