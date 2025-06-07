import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print(f"Retrieved GEMINI_API_KEY (first 5 chars): {GEMINI_API_KEY[:5]}*****")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        # Test a simple content generation
        print("Attempting a test content generation...")
        response = model.generate_content("Hello, world!")
        print("API test successful! Generated content:", response.text)
    except Exception as e:
        print(f"API test failed: {e}")
else:
    print("GEMINI_API_KEY is not set in environment variables.")