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
import flask
from flask import Blueprint, request, jsonify
from providers.provider_base import ProviderBase
from providers.provider_base import ProviderBase
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

api_bp = Blueprint("api", __name__)

# Define a mapping of supported providers to their API endpoints
PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    # Base URL, model appended later
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    # Requires local Ollama instance
    "ollama": "http://localhost:11434/api/generate",
    # Requires local LM Studio instance
    "lmstudio": "http://localhost:1234/v1/chat/completions",
}

# Credentials mapping for environment variables
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",  # Changed from GEMINI_API_KEY
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,  # Typically doesn't require an API key
    "lmstudio": None,  # Typically doesn't require an API key
}

# Map provider names to model families
PROVIDER_MODELS = {
    # User will need to define model as versions regularly and rapidly update.
    "openai": ["*"],
    "anthropic": ["*"],
    "gemini": ["*"],
    "openrouter": "*",
    "groq": ["*"],
    "mistral": ["*"],
    "ollama": ["*"],
    "lmstudio": ["*"],
}

# In-memory storage for PEA conversations
# Structure: {session_id: {'history': List[{'role': str, 'parts': List[{'text': str}]}], 'provider': str, 'model': str}}
pea_conversations: Dict[str, Dict] = {}

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


def initialize_pea_session(provider: str, model_name: Optional[str]) -> str:
    """Create a new PEA conversation session, storing provider and model."""
    session_id = str(uuid.uuid4())
    pea_conversations[session_id] = {
        "history": [],
        "provider": provider,
        "model": model_name,
    }
    return session_id


def add_message_to_session(
    session_id: str, role: str, content: str, provider: ProviderBase
) -> None:
    """Add a message to the conversation history in the provider's expected format."""
    if session_id in pea_conversations:
        # Use provider's message format method
        message = provider.format_message(role, content)
        pea_conversations[session_id]["history"].append(message)


def get_session_data(session_id: str) -> Optional[Dict]:
    """Retrieve the full session data (history, provider, model) for a session."""
    return pea_conversations.get(session_id)


def clear_session(session_id: str) -> None:
    """Remove a session and its history."""
    if session_id in pea_conversations:
        del pea_conversations[session_id]


# Removed redundant API key handling - moved to provider classes
def _get_provider_api_key(provider_name: str) -> Optional[str]:
    """Use provider-native key lookup via PROVIDER_KEYS env vars"""
    # DEPRECATED after provider-native implementation
    # Old value was PROVIDER_KEYS.get(provider_name)
    return None


def _generate_optimized_prompt_xml(
    provider: str, model_name: Optional[str], api_key: Optional[str], user_request: str
) -> str:
    """
    Generates an optimized XML prompt using the specified LLM provider.
    This function is designed for single-turn prompt optimization.
    """
    final_prompt_template = """
    Generate an optimized XML prompt based on the following user request.
    The XML structure and optimization approach should be determined
    by you to best fulfill the user's goal.
    Ensure the output is valid XML.

    User Request: {user_request}

    Provide the optimized prompt within <optimized_prompt></optimized_prompt> tags.
    """
    formatted_user_request = final_prompt_template.format(
        user_request=user_request
    ).strip()

    messages = [{"role": "user", "content": formatted_user_request}]
    return _generate_chat_response(provider, model_name, api_key, messages)


def _generate_chat_response(
    provider: str,
    model_name: Optional[str],
    api_key: Optional[str],
    chat_history: List[Dict],
) -> str:
    """
    Handles generating content through the appropriate API based on the provider,
    for ongoing chat conversations.
    """
    if provider == "google":
        genai.configure(api_key=api_key)
        # Use default model if none provided, or infer from context
        # (e.g., if a specific model was used to start the session)
        model_instance = genai.GenerativeModel(model_name=model_name or "gemini-pro")
        # Convert messages to Google's expected format: [{"role": ..., "parts": [{"text": ...}]}]
        formatted_history = []
        for msg in chat_history:
            if "content" in msg:
                # Convert OpenAI-style message to Google-style
                formatted_history.append(
                    {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                )
            else:
                # Already in Google format
                formatted_history.append(msg)

        # Use enum objects directly with type ignore to bypass strict type checking
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        # The type ignore comment will suppress the type checking error
        response = model_instance.generate_content(
            formatted_history, safety_settings=safety_settings  # type: ignore
        )
        return response.text.strip()

    elif provider in [
        "openai",
        "openrouter",
        "groq",
        "mistral",
        "ollama",
        "lmstudio",
        "anthropic",
    ]:
        headers = {"Content-Type": "application/json"}

        # Determine model name based on provider default if not specified
        if not model_name:
            default_models = {
                "openai": "gpt-4",
                "anthropic": "claude-3-opus-20240229",
                "groq": "llama3-8b-8192",
                "mistral": "mistral-large-latest",
                "openrouter": "google/gemini-pro",
                "ollama": "llama2",
                "lmstudio": "llama2",
            }
            model_name = default_models.get(provider, "default-model")
            logging.debug(
                f"Using default model '{model_name}' for provider '{provider}'"
            )

        # Add API key header based on provider
        if provider == "anthropic":
            if api_key:
                headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:  # OpenAI-compatible
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        # Special headers for OpenRouter
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://myprompt.alexjekop.com"
            headers["X-Title"] = "MyPrompt Assistant"

        url = PROVIDER_URLS.get(provider)
        if not url:
            raise ValueError(f"Unknown or unsupported provider URL for {provider}.")

        # Convert chat_history to the format expected by OpenAI-compatible APIs
        # which is [{'role': 'role_name', 'content': 'text_content'}]
        # The history passed by the PEA might have 'parts'
        formatted_messages = []
        for msg in chat_history:
            if "parts" in msg and isinstance(msg["parts"], list) and msg["parts"]:
                formatted_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["parts"][0][
                            "text"
                        ],  # Assuming only one text part
                    }
                )
            else:
                # Fallback for messages not in 'parts' format (e.g., initial user prompt)
                formatted_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "max_tokens": 4000,
        }
        # Ollama and LM Studio typically don't use 'temperature'
        # in /v1/chat/completions
        if provider not in ["ollama", "lmstudio"]:
            payload["temperature"] = (
                1  # Changed from 0.7 to 1 to satisfy type hint for 'int'
            )

        # Use a general timeout for requests, can be adjusted per provider
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        if provider == "anthropic":
            return response.json()["content"][0]["text"].strip()
        else:  # OpenAI-compatible structure
            return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@api_bp.route("/optimize-prompt", methods=["POST"])
async def optimize_prompt():
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
        # Check content type
        if not request.is_json:
            logging.error("optimize_prompt: Request must be JSON")
            return (
                jsonify({"error": "Unsupported Media Type: Must be application/json"}),
                415,
            )

        try:
            data = request.get_json()
        except Exception as e:
            logging.error(f"optimize_prompt: JSON parsing failed: {str(e)}")
            return jsonify({"error": "Invalid JSON format"}), 400

        logging.debug("optimize_prompt: Request JSON data received: {}".format(data))
        # Initial validation checks
        if not data:
            logging.error("optimize_prompt: No request body provided.")
            return jsonify({"error": "No request provided"}), 400
        if "request" not in data:
            logging.error("optimize_prompt: No request provided.")
            return jsonify({"error": "No request provided"}), 400
        if not data["request"]:
            logging.error("optimize_prompt: No request provided.")
            return jsonify({"error": "No request provided"}), 400

        user_request = data["request"]

        # Retrieve provider from Flask context (assuming it's set in before_request)
        selected_provider = await flask.g.providers.get(
            data.get("provider", "google").lower()
        )
        if not selected_provider:
            logging.error("optimize_prompt: Invalid provider specified")
            return jsonify({"error": "Invalid provider specified"}), 400

        model_name = data.get("model") or selected_provider._get_default_model()
        if not model_name:
            logging.error("optimize_prompt: Model name not provided and no default")
            return jsonify({"error": "Model name not provided and no default"}), 400

        logging.debug(
            "optimize_prompt: Selected provider: {}, Model: {}".format(
                selected_provider.name, model_name
            )
        )

        # Call the provider's generate_response with history
        messages = [{"role": "user", "content": user_request}]
        response_content = selected_provider.generate_content(messages)

        # Clean response and return
        optimized_prompt_xml = (
            response_content.replace("```xml", "").replace("```", "").strip()
        )
        return jsonify({"optimized_prompt": optimized_prompt_xml})

    except ValueError as e:
        logging.exception("optimize_prompt: ValueError caught.")
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"HTTP error: {e}"}), 500
    except Exception as e:
        logging.exception("optimize_prompt: Unexpected error caught.")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Temporary debug route in api.py
@api_bp.route("/pea/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def debug_pea_route(subpath):
    logging.debug("Debug PEA route hit for: {}".format(subpath))  # Removed f-string
    return jsonify({"message": "Debug PEA route hit for: {}".format(subpath)}), 200


@api_bp.route("/pea/start", methods=["POST"])
async def start_pea_session():
    """Initialize a new PEA conversation session and get the first response."""
    try:
        data = request.json
        if not data or "initial_request" not in data:
            return jsonify({"error": "No initial request provided"}), 400

        initial_request = data["initial_request"]
        # Extract provider and model from the request data
        provider = data.get("provider", "google").lower()  # Default to google
        model_name = data.get("model")  # Optional: user-selected model

        logging.debug(
            "start_pea_session: Provider: %s, Model: %s", provider, model_name
        )

        api_key = _get_provider_api_key(provider)
        if not api_key:
            return (
                jsonify(
                    {
                        "error": (
                            f"{PROVIDER_KEYS[provider]} or OPENAI_API_KEY not set "
                            "in environment for PEA mode."
                        )
                    }
                ),
                500,
            )

        # Create new session, storing provider and model
        session_id = initialize_pea_session(provider, model_name)

        # Add the system prompt and then the initial user request
        add_message_to_session(session_id, "system", PEA_SYSTEM_PROMPT)
        add_message_to_session(session_id, "user", initial_request)

        # Get PEA's first response (the model's turn)
        session_data = get_session_data(session_id)
        history = session_data["history"] if session_data else []

        # Pass the history to the generic chat response function
        pea_response_content = _generate_chat_response(
            provider, model_name, api_key, history
        )

        # Add PEA's response to history with role 'model'
        add_message_to_session(session_id, "model", pea_response_content)

        return jsonify({"session_id": session_id, "response": pea_response_content})

    except requests.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"error": f"Key error: {str(e)}"}), 500
    except TimeoutError as e:
        return jsonify({"error": f"Timeout error: {str(e)}"}), 500
    except Exception as e:
        logging.exception(f"start_pea_session: Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route("/pea/chat", methods=["POST"])
async def pea_chat():
    """Handle ongoing PEA conversation."""
    try:
        data = request.json
        if not data or "session_id" not in data or "message" not in data:
            return jsonify({"error": "Missing session_id or message"}), 400

        session_id = data["session_id"]
        user_message = data["message"]
        # Extract provider and model from the request data
        provider = data.get("provider", "google").lower()  # Default to google
        model_name = data.get("model")  # Optional: user-selected model

        logging.debug("pea_chat: Provider: %s, Model: %s", provider, model_name)

        # Verify session exists and get session data
        session_data = get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Invalid session ID"}), 400

        # Use provider and model from session if not provided in current request
        current_provider = data.get("provider", session_data["provider"]).lower()
        current_model_name = data.get("model", session_data["model"])

        logging.debug(
            "pea_chat: Provider: %s, Model: %s", current_provider, current_model_name
        )

        # Add user message to history
        add_message_to_session(session_id, "user", user_message)

        api_key = _get_provider_api_key(current_provider)
        if not api_key:
            return (
                jsonify(
                    {
                        "error": (
                            f"{provider.upper()}_API_KEY or OPENAI_API_KEY not set "
                            "in environment for PEA mode."
                        )
                    }
                ),
                500,
            )

        # Get PEA's response using the new helper
        # Re-fetch history after adding user message
        history = session_data["history"]
        pea_response_content = _generate_chat_response(
            current_provider, current_model_name, api_key, history
        )

        # Add PEA's response to history with role 'model'
        add_message_to_session(session_id, "model", pea_response_content)

        return jsonify({"response": pea_response_content})

    except requests.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"error": f"Key error: {str(e)}"}), 500
    except TimeoutError as e:
        return jsonify({"error": f"Timeout error: {str(e)}"}), 500
    except Exception as e:
        logging.exception(f"pea_chat: Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route("/pea/finalize", methods=["POST"])
async def finalize_prompt():
    """
    Finalizes the prompt generation process for a given session.

    This endpoint takes a JSON payload with a 'session_id' to identify the session.
    It verifies the session, and generates a final optimized prompt
    in XML format based on the conversation history using the selected provider/model.

    Returns:
        JSON response containing the final optimized prompt or an error message.
    """
    try:
        data = request.json
        if not data or "session_id" not in data:
            return jsonify({"error": "No session_id provided"}), 400

        session_id = data["session_id"]
        # Extract provider and model from the request data
        provider = data.get("provider", "google").lower()  # Default to google
        model_name = data.get("model")  # Optional: user-selected model

        logging.debug("finalize_prompt: Provider: %s, Model: %s", provider, model_name)

        # Verify session exists and get session data
        session_data = get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Invalid session ID"}), 400

        # Use provider and model from session if not provided in current request
        current_provider = data.get("provider", session_data["provider"]).lower()
        current_model_name = data.get("model", session_data["model"])

        logging.debug(
            "finalize_prompt: Provider: %s, Model: %s",
            current_provider,
            current_model_name,
        )

        api_key = _get_provider_api_key(current_provider)
        if not api_key:
            return (
                jsonify(
                    {
                        "error": (
                            f"{provider.upper()}_API_KEY or OPENAI_API_KEY not set "
                            "in environment for PEA mode."
                        )
                    }
                ),
                500,
            )

        # Add finalization request/instruction to history (user role)
        finalize_instruction = """Based on our conversation, please generate the final optimized prompt in XML format.
            The XML should capture all the key information we've discussed and be structured to
            effectively achieve the user's goal."""
        add_message_to_session(session_id, "user", finalize_instruction)

        # Get final XML prompt using the new helper
        history = session_data["history"]  # Re-fetch history after adding instruction
        final_prompt_content = _generate_chat_response(
            current_provider, current_model_name, api_key, history
        )

        # Clean any markdown code block wrappers from the response
        final_prompt_content = (
            final_prompt_content.replace("```xml", "").replace("```", "").strip()
        )

        # Clean up the session
        clear_session(session_id)

        return jsonify({"final_prompt": final_prompt_content})

    except requests.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"error": f"Key error: {str(e)}"}), 500
    except TimeoutError as e:
        return jsonify({"error": f"Timeout error: {str(e)}"}), 500
    except Exception as e:
        logging.exception(f"finalize_prompt: Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500