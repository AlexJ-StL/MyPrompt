from app_factory import create_app
import logging

app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=5001)
