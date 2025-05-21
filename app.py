from flask import Flask
from flask_cors import CORS
from api import api_bp

app = Flask(__name__)
CORS(app) # Enable CORS for all origins

app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    """
    Handles the root URL request and returns a welcome message.

    Parameters:
    None

    Returns:
    str: A welcome message indicating the application name.

    Exceptions:
    None
    """
    return "MyPrompt Backend"

if __name__ == '__main__':
    app.run(debug=True)