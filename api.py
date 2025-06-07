"""
This module provides API endpoints for optimizing prompts using various language models.
It includes functionality for starting, continuing, and finalizing prompt engineering
assistant (PEA) sessions, as well as handling errors and exceptions.
"""

import os
import uuid
from typing import Dict, List, Optional
import logging
import requests
from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

api_bp = Blueprint('api', __name__)

# Define a mapping of supported providers to their API endpoints
PROVIDER_URLS = {
    'openai': 'https://api.openai.com/v1/chat/completions',
    'anthropic': 'https://api.anthropic.com/v1/messages',
    # Base URL, model appended later
    'google': 'https://generativelanguage.googleapis.com/v1beta/models',
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'groq': 'https://api.groq.com/openai/v1/chat/completions',
    'mistral': 'https://api.mistral.ai/v1/chat/completions',
    # Requires local Ollama instance
    'ollama': 'http://localhost:11434/api/generate',
    # Requires local LM Studio instance
    'lmstudio': 'http://localhost:1234/v1/chat/completions'
}

# Credentials mapping for environment variables
PROVIDER_KEYS = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'gemini': 'GEMINI_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'groq': 'GROQ_API_KEY',
    'ollama': None,  # Typically doesn't require an API key
    'lmstudio': None  # Typically doesn't require an API key
}

# Map provider names to model families
PROVIDER_MODELS = {
    #User will need to define model as versions regularly and rapidly update.
    'openai': ['*'],
    'anthropic': ['*'],
    'gemini': ['*'],
    'openrouter': '*',
    'groq': ['*'],
    'mistral': ['*'],
    'ollama': ['*'],
    'lmstudio': ['*']
}

# In-memory storage for PEA conversations
# Structure: {session_id: List[{'role': str, 'parts': List[{'text': str}]}]}
pea_conversations: Dict[str, List[dict]] = {}

# PEA system prompt
PEA_SYSTEM_PROMPT = """# System Prompt: Prompt Engineering Assistant (PEA)

## Your Role:
You are PEA, an AI Prompt Engineering Assistant, operating with the role 'model' in this conversation. Your job is to help the user create the best possible prompt by asking questions and gathering information.

## Initial Task - VERY IMPORTANT:
Your ONLY goal in your FIRST response to the user's initial request is to ask a clarifying question to begin the information gathering process. Do NOT provide the final XML prompt, an example of the final XML, or any structured output in your first response. Just ask a question to start the conversation based on the user's request and the Key Areas to Probe.

## Core Process:
1. Understand Initial Request.
2. Analyze for Completeness & Clarity (using Key Areas to Probe).
3. Engage in Iterative Questioning & Clarification: Ask targeted questions, one or a few at a time. Explain why information is needed.
4. Synthesize and Structure: ONLY when the user indicates they are finished (by triggering a separate command), generate the final, optimized prompt in the specified XML format.

## Key Areas to Probe:
- Task Definition & Objective (goal, desired outcome, target audience)
- Context & Background (domain knowledge, terminology)
- LLM Persona/Role
- Input/Output Requirements (format, structure, examples)
- Constraints & Rules
- Technical Specifications (if applicable)
- Edge Cases & Error Handling

## Interaction Style:
- Be collaborative and patient.
- Explain your reasoning when asking for certain types of information.
- Assume the user has a clear goal but may not know how to best articulate it for an LLM.
- Prioritize clarity and precision.
- Do not make assumptions. If something is unclear, ask.

## Final Output Requirement (Applies ONLY when specifically requested by the user after the conversation):
When the user signals they are finished and require the final prompt, you will generate it in valid XML format. The XML structure should effectively organize the information gathered during our conversation to create a comprehensive prompt for another LLM.
"""

def initialize_pea_session() -> str:
    """Create a new PEA conversation session."""
    session_id = str(uuid.uuid4())
    pea_conversations[session_id] = []
    return session_id

def add_message_to_session(session_id: str, role: str, content: str) -> None:
    """Add a message to the conversation history in the correct format for the API."""
    if session_id in pea_conversations:
        pea_conversations[session_id].append({
            'role': role,
            'parts': [{'text': content}]
        })

def get_session_history(session_id: str) -> Optional[List[dict]]:
    """Retrieve the conversation history for a session."""
    return pea_conversations.get(session_id)

def clear_session(session_id: str) -> None:
    """Remove a session and its history."""
    if session_id in pea_conversations:
        del pea_conversations[session_id]

def _get_provider_api_key(provider_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given provider from environment variables.
    Handles fallback to OPENAI_API_KEY for OpenAI-compatible providers.
    """
    api_key_env_var = f"{provider_name.upper()}_API_KEY"
    api_key = os.getenv(api_key_env_var)

    # Specific fallback for OpenAI-compatible providers
    if not api_key and provider_name in ['openrouter', 'groq', 'mistral',
                                         'ollama', 'lmstudio']:
        api_key = os.getenv('OPENAI_API_KEY')
    return api_key

def _generate_content_through_api(
    provider: str, model_name: Optional[str], api_key: str, user_request: str
) -> str:
    """
    Handles generating content through the appropriate API based on the provider.
    This function consolidates the API call logic for different LLM providers.
    """
    final_prompt_template = """
    Generate an optimized XML prompt based on the following user request.
    The XML structure and optimization approach should be determined
    by you to best fulfill the user's goal.
    Ensure the output is valid XML.

    User Request: {user_request}

    Provide the optimized prompt within <optimized_prompt></optimized_prompt> tags.
    """
    formatted_user_request = final_prompt_template.format(user_request=user_request).strip()

    if provider == 'google':
        genai.configure(api_key=api_key)
        # Use default model if none provided
        model_instance = genai.GenerativeModel(
            model_name=model_name or "gemini-pro"  # Changed from "gemini-2.5-flash"
        )
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        response = model_instance.generate_content(
            formatted_user_request, safety_settings=safety_settings
        )
        return response.text.strip()

    elif provider in ['openai', 'openrouter', 'groq', 'mistral',
                      'ollama', 'lmstudio', 'anthropic']:
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Determine model name based on provider default if not specified
        if not model_name:
            default_models = {
                'openai': "gpt-4",
                'anthropic': "claude-3-opus-20240229",
                'groq': "llama3-8b-8192",
                'mistral': "mistral-large-latest",
                'openrouter': "google/gemini-pro",
                'ollama': "llama2",
                'lmstudio': "llama2",
            }
            model_name = default_models.get(provider, "default-model")

        # Add API key header based on provider
        if provider == 'anthropic':
            headers['x-api-key'] = api_key
            headers['anthropic-version'] = '2023-06-01'
        else: # OpenAI-compatible
            headers['Authorization'] = f'Bearer {api_key}'

        # Special headers for OpenRouter
        if provider == 'openrouter':
            headers['HTTP-Referer'] = 'https://myprompt.alexjekop.com'
            headers['X-Title'] = 'MyPrompt Assistant'

        url = PROVIDER_URLS.get(provider)
        if not url:
            raise ValueError(f"Unknown or unsupported provider URL for {provider}.")

        messages = [{"role": "user", "content": formatted_user_request}]

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 4000
        }
        # Ollama and LM Studio typically don't use 'temperature'
        # in /v1/chat/completions
        if provider not in ['ollama', 'lmstudio']:
            payload["temperature"] = 0.7


        # Use a general timeout for requests, can be adjusted per provider
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        if provider == 'anthropic':
            return response.json()["content"][0]["text"].strip()
        else: # OpenAI-compatible structure
            return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@api_bp.route('/optimize-prompt', methods=['POST'])
def optimize_prompt():
    """
    Optimizes a user's prompt by sending it to a selected language model provider.

    This endpoint accepts a JSON payload with a 'request' (the user's prompt)
    and optional 'provider' (default: 'google') and 'model' parameters.
    It retrieves the necessary API key, dispatches the request to the
    appropriate LLM provider, and returns the optimized prompt in XML format.

    Returns:
        JSON response containing the optimized prompt in XML or an error message.
    """
    logging.debug("optimize_prompt: Function entered.")
    try:
        data = request.json
        logging.debug("optimize_prompt: Request JSON data received: {}".format(data))
        # Initial validation checks
        if not data:
            logging.error("optimize_prompt: No request body provided.")
            return jsonify({"error": "Request body must be JSON"}), 400
        if 'request' not in data or not data['request']:
            logging.error("optimize_prompt: Missing or empty 'request' field.")
            return jsonify({"error": "No 'request' field provided in JSON"}), 400

        user_request = data['request']
        # Default to Google, ensure lowercase
        provider = data.get('provider', 'google').lower()
        model_name = data.get('model') # Optional: model name for the provider
        logging.debug("optimize_prompt: Provider: {}, Model: {}".format(
            provider, model_name))

        # Retrieve API key
        api_key = _get_provider_api_key(provider)
        # Log the API key value (first 5 chars for security)
        logging.debug("optimize_prompt: Retrieved API key (first 5 chars): %s",
                      api_key[:5] if api_key else "None/Empty")

        if not api_key:
            logging.error("optimize_prompt: API key not set for provider: {}".format(
                provider))
            return jsonify(
                {"error": (f"{provider.upper()}_API_KEY or OPENAI_API_KEY not set "
                           "in environment")}
            ), 500
        logging.debug("optimize_prompt: API key retrieved successfully.")

        # Call the appropriate external API helper
        logging.debug("optimize_prompt: Calling _generate_content_through_api...")
        optimized_prompt_xml = _generate_content_through_api(
            provider, model_name, api_key, user_request
        )
        logging.debug("optimize_prompt: _generate_content_through_api returned successfully.")

        # Clean any markdown code block wrappers from the response
        optimized_prompt_xml = (optimized_prompt_xml
                                .replace("```xml", "")
                                .replace("```", "")
                                .strip())
        
        logging.debug("optimize_prompt: Returning optimized prompt.")
        return jsonify({"optimized_prompt": optimized_prompt_xml})

    except ValueError as e:
        logging.exception("optimize_prompt: ValueError caught.")
        # Catch errors from _generate_content_through_api or other logic
        # 400 for bad request like unsupported provider or invalid model
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        logging.exception("optimize_prompt: HTTPError caught.")
        return jsonify({"error": f"Provider API returned an error: {e.response.text}"}), \
            status_code
    except requests.RequestException as e:
        logging.exception("optimize_prompt: RequestException caught.")
        return jsonify(
            {"error": (f"Failed to communicate with provider API due to network or "
                       f"request issue: {str(e)}")}
        ), 500
    except KeyError as e:
        logging.exception("optimize_prompt: KeyError caught.")
        return jsonify({"error": f"Unexpected API response format: Missing key {str(e)}"}), 500
    except TimeoutError as e:
        logging.exception("optimize_prompt: TimeoutError caught.")
        return jsonify({"error": f"API call timed out: {str(e)}"}), 500
