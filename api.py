from flask import Blueprint, request, jsonify
import os
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

api_bp = Blueprint('api', __name__)

@api_bp.route('/optimize-prompt', methods=['POST'])
def optimize_prompt():
    data = request.json
    if not data or 'request' not in data:
        return jsonify({"error": "No request provided"}), 400

    user_request = data.get('request')

    if not user_request:
        return jsonify({"error": "No request provided"}), 400

    # Use os.getenv('GEMINI_API_KEY') to get the API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    genai.configure(api_key=api_key)

    # Configure safety settings (example: block dangerous content)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        # Initialize the Generative Model
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-04-17",
            safety_settings=safety_settings
        )

        # Construct the prompt for the LLM
        # Instruct the LLM to generate an optimized XML prompt based on the user's request.
        # The LLM should determine the best approach for optimization and the XML structure.
        llm_prompt = f"""
        Generate an optimized XML prompt based on the following user request.
        The XML structure and optimization approach should be determined by you to best fulfill the user's goal.
        Ensure the output is valid XML.

        User Request: {user_request}

        Provide the optimized prompt within <optimized_prompt></optimized_prompt> tags.
        """

        # Make the API call
        response = model.generate_content(llm_prompt)

        # Extract the XML content from the response
        # Assuming the LLM response contains the XML within the specified tags
        # This might need more robust parsing depending on actual LLM output
        optimized_prompt_xml = response.text.strip()

        return jsonify({"optimized_prompt": optimized_prompt_xml})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# TODO: Add unit tests for this API endpoint