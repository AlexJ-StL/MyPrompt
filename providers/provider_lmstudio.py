from providers.provider_base import ProviderBase


class LMStudioProvider(ProviderBase):
    API_KEY_ENV = None

    def __init__(self):
        pass

    def call(self, message):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": message.model,
            "messages": [{"role": "user", "content": message.content}],
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except Exception as e:
            logging.error(f"LM Studio provider call failed: {e}")
            return {"Error": str(e)}
