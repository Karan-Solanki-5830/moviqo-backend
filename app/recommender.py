"""Recommendation logic, state management, and data loading."""

import pandas as pd
import numpy as np
import ast
import joblib
from collections import defaultdict
from scipy.sparse import issparse, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from app.config import MODELS_DIR, DATA_DIR, EPSILON
from app.database import get_user_base_genres, get_user_interactions

# --- Global State (Loader) ---

df = None
tfidf_primary = None
tfidf_secondary = None
movie_content_matrix = None
svd_scores = None
user_map = None


def list_to_text(x):
    """Convert list-like strings to text."""
    if not isinstance(x, str):
        return str(x)
    x = x.strip()
    if x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return " ".join(map(str, parsed))
        except:
            pass
    return x


def load_all():
    """Load all models and data into global state."""
    global df, tfidf_primary, tfidf_secondary, movie_content_matrix, svd_scores, user_map
    
    print("Loading data from CSV...")
    csv_path = DATA_DIR / "FINAL_CSV_TO_BE_USED.csv"
    df_full = pd.read_csv(csv_path)
    
    print("Loading SVD indices...")
    svd_movie_indices = joblib.load(MODELS_DIR / "svd_movie_indices.joblib")
    
    df = df_full.iloc[svd_movie_indices].reset_index(drop=True)
    
    df["genre_keyword_text"] = (
        df["genres"].apply(list_to_text) + " " +
        df["keywords"].apply(list_to_text)
    ).str.lower()
    
    df["overview_text"] = df["overview"].fillna("").str.lower()
    
    print("Loading TF-IDF models...")
    tfidf_primary = joblib.load(MODELS_DIR / "tfidf_primary.joblib")
    tfidf_secondary = joblib.load(MODELS_DIR / "tfidf_secondary.joblib")
    
    print("Building content matrix...")
    matrix_primary = tfidf_primary.transform(df["genre_keyword_text"])
    matrix_secondary = tfidf_secondary.transform(df["overview_text"])
    
    movie_content_matrix = hstack([
        matrix_primary * 0.7,
        matrix_secondary * 0.3
    ])
    assert issparse(movie_content_matrix), "movie_content_matrix must be sparse"
    
    print("Loading SVD scores...")
    svd_scores = np.load(MODELS_DIR / "svd_predicted_scores.npy")
    user_ids = joblib.load(MODELS_DIR / "user_ids.joblib")
    
    user_map = {u: i for i, u in enumerate(user_ids)}
    
    print(f"Loaded {len(df)} movies and {len(user_ids)} users. Content Matrix Shape: {movie_content_matrix.shape}")
    return True


# --- Baseline Logic ---

def genre_baseline_scores(user_genres):
    """Calculate baseline scores based on user genres."""
    assert len(user_genres) >= 3, "Expected at least 3 base genres"
    base_genres = [g.lower() for g in user_genres[:3]]
    query_text = " ".join(base_genres)
    
    q_vec_primary = tfidf_primary.transform([query_text])
    q_vec_secondary = tfidf_secondary.transform([""])
    
    q_vec_combined = hstack([
        q_vec_primary * 0.7,
        q_vec_secondary * 0.3
    ])

    scores = cosine_similarity(q_vec_combined, movie_content_matrix).flatten()
    assert len(scores) == len(df), "baseline scores must match number of movies"
    return scores


# --- Picker Logic ---

def pick_movie(scores, k=15):
    """Pick a movie index from scores using diverse/best strategy."""
    top = np.argsort(scores)[-k:]
    diverse = np.random.choice(top[:k//2])
    best = np.random.choice(top[k//2:])
    return int(np.random.choice([diverse, best]))


# --- Preference Logic ---

class UserPreferenceState:
    """
    Tracks and updates user genre preferences over time.
    Values increase with 'likes' and decrease with 'dislikes'.
    used to re-rank recommendation candidates.
    """
    def __init__(self, initial_genres, base_weight=1.0, min_weight=0.2):
        self.genre_weights = defaultdict(lambda: min_weight)
        self.min_weight = min_weight
        self.history = []

        for g in initial_genres:
            self.genre_weights[g.lower()] = base_weight

    def update(self, movie_genres, action):
        """
        Update weights based on user feedback.
        Like: +0.3
        Dislike: -0.25 (floor at min_weight)
        """
        if action == "like":
            delta = 0.3
        elif action == "dislike":
            delta = -0.25
        else:
            return

        for g in movie_genres:
            g = g.lower()
            self.genre_weights[g] = max(
                self.min_weight,
                self.genre_weights[g] + delta
            )

        self.history.append((movie_genres, action))

    def score_movie(self, movie_genres):
        """
        Calculate a total preference score for a movie
        by summing the current weights of its genres.
        """
        score = 0.0
        for g in movie_genres:
            score += self.genre_weights[g.lower()]
        return score


def build_preference_state(interactions, base_genres):
    """
    Replay user history to rebuild their current preference state.
    Starts with base genres, then applies all past interactions in order.
    """
    pref = UserPreferenceState(base_genres)
    
    for movie_index, action in interactions:
        genres_str = df.loc[movie_index, "genres"]
        try:
            genres = ast.literal_eval(genres_str)
        except:
            continue
            
        pref.update(genres, action)
    
    return pref


# --- Core Engine Logic ---

def normalize(x):
    """Normalize a vector to range [0, 1]."""
    x = np.array(x)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def get_next_movie(db: Session, user_id: int) -> dict:
    """
    Core function to select the next movie recommendation.
    
    1. Fetches user data (base genres, interactions).
    2. Builds preference state from history.
    3. Calculates Hybrid Score:
       - 70% Content-Based (TF-IDF similarity to base genres)
       - 30% SVD (Collaborative Filtering prediction)
    4. Applies Re-ranking buffer based on live feedback (likes/dislikes).
    5. Filters out already seen movies.
    6. Selects movie using Epsilon-Greedy Exploration to discover new interests.
    """

    # 1. & 2. Get User Data & Build Preference State
    base_genres = get_user_base_genres(db, user_id)
    if base_genres is None:
        raise ValueError(f"User {user_id} not found")
    
    interactions = get_user_interactions(db, user_id)
    pref_state = build_preference_state(interactions, base_genres)
    
    # 3. Calculate Hybrid Scores

    # Content-based baseline (TF-IDF)
    content_scores = genre_baseline_scores(base_genres)
    
    # SVD Collaborative Filtering Score
    if user_id in user_map:
        row_idx = user_map[user_id]
        svd_user_scores = svd_scores[row_idx]
    else:
        svd_user_scores = np.zeros(len(df))
        
    norm_content = normalize(content_scores)
    norm_svd = normalize(svd_user_scores)
    
    # Weighted Hybrid Combination
    hybrid_scores = 0.7 * norm_content + 0.3 * norm_svd
    
    # 4. Re-Ranking with Preference Feedback
    beta = 0.3  # Influence of preference boost
    
    final_scores = np.zeros_like(hybrid_scores)
    
    for i in range(len(df)):
        genres_str = df.loc[i, "genres"]
        try:
            genres = ast.literal_eval(genres_str)
        except:
            genres = []
            
        pref_score = pref_state.score_movie(genres)
        
        # Boost hybrid score by preference multiplier
        final_scores[i] = hybrid_scores[i] * (1 + beta * pref_score)
        
    # 5. Filter Seen Movies
    seen = {movie_index for movie_index, _ in interactions}
    for i in seen:
        final_scores[i] = -np.inf
        
    step = len(interactions)
    
    # 6. Epsilon-Greedy Strategy

    # First 2 steps: Pure exploitation (best matches)
    if step < 2:
        current_eps = 0.0 
    else:
        # Increase exploration as user interacts more
        current_eps = min(EPSILON, 0.15 + 0.02 * step)
        
    if np.random.rand() < current_eps:
        # Explore: Pick random from top 400 candidates
        top_k = min(400, len(df))
        top_candidates = np.argsort(final_scores)[-top_k:]
        nxt = int(np.random.choice(top_candidates))
    else:
        # Exploit: Pick best available movie using softmax picker
        nxt = int(pick_movie(final_scores))
        
    rec = df.loc[nxt]
    
    # Helper to safely extract values from movie record
    def get_safe(key, default=None, cast=None):
        val = rec.get(key, default)
        if pd.isna(val) or val == "nan":
            return default
        if cast:
            try:
                return cast(val)
            except:
                return default
        return val

    # 7. Format Output
    return {
        "movie_index": int(nxt),
        "title": str(rec["title"]),
        "genres": rec["genres"],
        "overview": get_safe("overview", "No overview available."),
        "poster_path": get_safe("poster_path"),
        "imdb_id": get_safe("imdb_id"),
        "vote_average": get_safe("vote_average", cast=float),
        "runtime": get_safe("runtime", cast=int),
        "release_date": get_safe("release_date"),
        "original_language": get_safe("original_language"),
        "spoken_languages": get_safe("spoken_languages"),
        "keywords": get_safe("keywords"),
        "revenue": get_safe("revenue", cast=float),
        "hybrid_score": float(hybrid_scores[nxt])
    }
