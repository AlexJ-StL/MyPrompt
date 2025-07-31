from providers.provider_base import ProviderBase
from typing import List, Dict
import requests
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
import logging

class GoogleProvider(ProviderBase):
    key_env_var = "GOOGLE_API_KEY"

    def _get_default_model(self) -> str:
        return "gemini-1.5-flash"

    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        # Google's API uses a different format for messages
        formatted_history = []
        for msg in chat_history:
            if "content" in msg:
                formatted_history.append(
                    {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                )
            else:
                formatted_history.append(msg)
        return formatted_history

    def _make_request(self, url: str, data: Dict) -> requests.Response:
        # This method is not directly used by generate_content in this provider
        # but is required by the abstract base class.
        # We'll implement a placeholder that raises NotImplementedError
        raise NotImplementedError("GoogleProvider uses direct API calls instead of _make_request")

    def generate_content(self, chat_history: List[Dict]) -> str:
        # Configure the Google API
        genai.configure(api_key=self.api_key)
        
        # Create model instance
        model_instance = genai.GenerativeModel(model_name=self.model_name or self._get_default_model())
        
        # Format the history
        formatted_history = self.format_messages(chat_history)

        # Define safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Generate content
        try:
            response = model_instance.generate_content(
                formatted_history, safety_settings=safety_settings
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error in GoogleProvider.generate_content: {e}")
            raise