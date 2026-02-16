"""Flask application factory."""

from flask import Flask
from flask_cors import CORS

from app.config import config
from app.database import init_db
from app.routes import api_bp
from app.recommender import load_all


def create_app(config_name="development"):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    

    CORS(app, origins=app.config.get("CORS_ORIGINS"))
    

    init_db()
    

    print("Loading models and data...")
    load_all()
    print("Ready!")
    

    app.register_blueprint(api_bp)
    

    @app.route("/")
    def index():
        return {"message": "Moviqo API", "version": "1.0.0"}
    
    return app
