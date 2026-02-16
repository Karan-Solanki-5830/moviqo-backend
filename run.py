#!/usr/bin/env python
"""
Entry point to run the Flask application.

Local Development:
    python run.py

Production (Gunicorn):
    gunicorn run:app
"""

import os
import sys

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_ENV") != "production"
    )
