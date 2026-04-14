# moviqo-backend

Flask backend for Moviqo. Handles movie data, recommendations, and serves a REST API consumed by the frontend.

## Stack

- Python / Flask
- SQLAlchemy + PostgreSQL
- scikit-learn (for recommendations)
- Gunicorn (production)

## Getting Started

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

DATABASE_URL=postgresql://user:password@localhost/moviqo
FLASK_ENV=development
SECRET_KEY=your-secret-key

Then run:

```bash
python run.py
```

API will be available at `http://localhost:5000`.

## Production

```bash
gunicorn run:app
```

## Project Structure
app/      # Flask app factory and routes
models/   # SQLAlchemy models
data/     # Seed data or static datasets
run.py    # Entry point

