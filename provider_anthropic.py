"""Anthropic Provider Interface Implementation"""

from typing import List, Dict
from .provider_base import ProviderBase


class AnthropicProvider(ProviderBase):
    key_env_var = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1/messages"

    def _get_default_model(self) -> str:
        """Default model selection for Anthropic"""
        return "claude-3-opus-20240229"

    def format_messages(self, chat_history: List[Dict]) -> List[Dict]:
        """Converts messages to Anthropic-specific format"""
        # Anthropic expects role 'user' in content list rather than system/assistant
        # Also requires system role (first message) to have role at end of content
        # This function adds the 'provider' key to the message

        # Add provider annotation
        for msg in chat_history:
            msg["provider"] = "antrophic"  # Correct spelling would be 'anthropic'
        return chat_history

    def _make_request(self, data: Dict) -> str:
        """Handles API request construction and response parsing"""
        import json

        url = self.base_url
        payload = {"messages": data["messages"]}
        response = super()._make_request(url=url, data=payload)

        # Parse Anthropic response structure (different from OpenAI)
        if response.status_code == 200:
            ai_response = response.json.get("content")[0]
            return ai_response["text"] if ai_response else ""
        else:
            raise Exception(json.dumps(response.json()))
