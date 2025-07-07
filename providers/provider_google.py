from providers.provider_base import ProviderBase


class GoogleProvider(ProviderBase):
    API_KEY_ENV = "GOOGLE_API_KEY"

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("The API key is required to create a Google provider.")
        self.api_key = api_key

    def call(self, message):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/generateContent"
        data = message.convert()
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        except Exception as e:
            logging.error(f"Google provider call failed: {e}")
            return {"Error": str(e)}
