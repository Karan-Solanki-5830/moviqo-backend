"""Application configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent.resolve()
APP_DIR = Path(__file__).parent.resolve()

# Data and model paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# Recommender constants
EPSILON = 0.35  # Max exploration rate
ALPHA = 0.7     # Hybrid weight

# Flask settings
class Config:
    """Flask configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Fix for SQLAlchemy requiring 'postgresql://' instead of 'postgres://'
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        
    SQLALCHEMY_DATABASE_URI = database_url
    try:
        from sqlalchemy.engine.url import make_url
        url = make_url(database_url)
        print(f"Connected to database host: {url.host}")
    except Exception:
        print("Connected to database (host unknown)")
        
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # CORS Configuration
    frontend_url = os.environ.get("FRONTEND_URL")
    
    # In production, FRONTEND_URL is mandatory
    if os.environ.get("FLASK_ENV") == "production" and not frontend_url:
        raise ValueError("FRONTEND_URL environment variable is required in production")
        
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    if frontend_url:
        CORS_ORIGINS.append(frontend_url)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
