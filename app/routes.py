"""API route definitions."""

import ast
import pandas as pd
from flask import Blueprint, request, jsonify
from app.database import get_session, create_user, user_exists, add_interaction, get_user_history
from app import recommender
from app.recommender import get_next_movie

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/users", methods=["POST"])
def register_user():
    """Create a new user with base genres."""
    data = request.json
    base_genres = data.get("base_genres")
    
    if not base_genres or not isinstance(base_genres, list) or len(base_genres) != 3:
        return jsonify({"error": "base_genres must be a list of exactly 3 genres"}), 400
    
    db = get_session()
    try:
        user_id = create_user(db, base_genres)
        return jsonify({"user_id": user_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        db.close()


@api_bp.route("/recommend", methods=["GET"])
def recommend_movie():
    """Get next movie recommendation for a user."""
    user_id = request.args.get("user_id", type=int)
    
    if user_id is None:
        return jsonify({"error": "user_id parameter required"}), 400
    
    db = get_session()
    try:
        if not user_exists(db, user_id):
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        movie = get_next_movie(db, user_id)
        return jsonify(movie), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@api_bp.route("/feedback", methods=["POST"])
def submit_feedback():
    """Store user feedback (like/dislike) for a movie."""
    data = request.json
    user_id = data.get("user_id")
    movie_index = data.get("movie_index")
    action = data.get("action")
    
    if user_id is None or movie_index is None or action is None:
        return jsonify({"error": "user_id, movie_index, and action required"}), 400
    
    if action not in ["like", "dislike"]:
        return jsonify({"error": "action must be 'like' or 'dislike'"}), 400
    
    db = get_session()
    try:
        if not user_exists(db, user_id):
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        add_interaction(db, user_id, movie_index, action)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        db.close()


@api_bp.route("/history", methods=["GET"])
def get_history():
    """Get full interaction history for a user."""
    user_id = request.args.get("user_id", type=int)
    
    if user_id is None:
        return jsonify({"error": "user_id parameter required"}), 400
    
    db = get_session()
    try:
        if not user_exists(db, user_id):
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        history_data = get_user_history(db, user_id)
        
        # Enrich with movie details
        enriched_history = []
        if recommender.df is None:
            print("ERROR: recommender.df is None!")
            raise Exception("Recommendations not loaded")
            
        print(f"DEBUG: Processing {len(history_data)} items. df shape: {recommender.df.shape}")
        
        for item in history_data:
            idx = item["movie_index"]
            if idx in recommender.df.index:
                movie_row = recommender.df.loc[idx]
                
                # Parse genres safely
                try:
                    genres = ast.literal_eval(movie_row["genres"])
                except:
                    genres = []

                item["title"] = movie_row["title"]
                item["genres"] = genres
                
                # Handle poster path
                poster = movie_row.get("poster_path", None)
                if pd.isna(poster) or poster == "nan":
                    poster = None
                item["poster_path"] = poster
            else:
                item["title"] = "Unknown Title"
                item["genres"] = []
                item["poster_path"] = None
                
            enriched_history.append(item)
            
        return jsonify(enriched_history), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200
