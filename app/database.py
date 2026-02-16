"""Database setup, models, and CRUD operations."""

import json
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session


# --- Database Setup ---

# Create SQLite engine with proper path handling
from app.config import Config

# Create engine from config
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Alias for compatibility
db = Base

def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_session():
    """Get a new database session. Caller must close it."""
    return SessionLocal()


# --- Models ---

class User(Base):
    """User model storing base genre preferences."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    base_genres = Column(String, nullable=False)  # JSON string of 3 genres
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to interactions
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, base_genres={self.base_genres})>"


class Interaction(Base):
    """User interaction model for likes/dislikes."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_index = Column(Integer, nullable=False)
    action = Column(String, nullable=False)  # "like" or "dislike"
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to user
    user = relationship("User", back_populates="interactions")

    def __repr__(self):
        return f"<Interaction(user_id={self.user_id}, movie_index={self.movie_index}, action={self.action})>"


# --- CRUD Operations ---

def create_user(db: Session, base_genres: List[str]) -> int:
    """Create a new user with base genres. Returns user_id."""
    if len(base_genres) != 3:
        raise ValueError("base_genres must contain exactly 3 genres")
    
    user = User(base_genres=json.dumps(base_genres))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user.id


def get_user_base_genres(db: Session, user_id: int) -> Optional[List[str]]:
    """Get user's base genres. Returns None if user not found."""
    # Find user by ID
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None
    return json.loads(user.base_genres)


def get_user_interactions(db: Session, user_id: int) -> List[Tuple[int, str]]:
    """Get all interactions for a user. Returns list of (movie_index, action) tuples."""
    interactions = db.query(Interaction).filter(
        Interaction.user_id == user_id
    ).order_by(Interaction.timestamp).all()
    
    return [(inter.movie_index, inter.action) for inter in interactions]


def add_interaction(db: Session, user_id: int, movie_index: int, action: str) -> None:
    """Add a new interaction."""
    if action not in ["like", "dislike"]:
        raise ValueError("action must be 'like' or 'dislike'")
    
    interaction = Interaction(
        user_id=user_id,
        movie_index=movie_index,
        action=action
    )
    print(f"DEBUG: Adding interaction: user={user_id}, movie={movie_index}, action={action}")
    db.add(interaction)
    db.commit()
    print("DEBUG: Interaction committed.")


def get_user_history(db: Session, user_id: int) -> List[dict]:
    """Get full interaction history for a user."""
    interactions = db.query(Interaction).filter(
        Interaction.user_id == user_id
    ).order_by(Interaction.timestamp).all()
    
    print(f"DEBUG: get_user_history retrieved {len(interactions)} interactions for user {user_id}")
    
    return [
        {
            "id": inter.id,
            "movie_index": inter.movie_index,
            "action": inter.action,
            "timestamp": inter.timestamp.isoformat()
        }
        for inter in interactions
    ]


def user_exists(db: Session, user_id: int) -> bool:
    """Check if user exists."""
    return db.query(User).filter(User.id == user_id).first() is not None
