from providers.provider_base import ProviderBase


class OllamaProvider(ProviderBase):
    API_KEY_ENV = None

    def __init__(self):
        pass

    def call(self, message):
        url = f"{self.base_url}/{message.model}/generate"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": message.content}

        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except Exception as e:
            logging.error(f"Ollama provider call failed: {e}")
            return {"Error": str(e)}
