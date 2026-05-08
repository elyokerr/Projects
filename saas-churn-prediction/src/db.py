"""
Database connection utilities.

Provides a SQLAlchemy engine and session management.
Supports both PostgreSQL (Docker) and SQLite (local dev) via config toggle.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import ACTIVE_DB_URL, USE_POSTGRES


def get_engine():
    """Create and return a SQLAlchemy engine based on active config."""
    connect_args = {}
    if not USE_POSTGRES:
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        ACTIVE_DB_URL,
        echo=False,
        connect_args=connect_args,
    )
    return engine


def get_session():
    """Create a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection():
    """Verify database connectivity."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✓ Database connection successful ({ACTIVE_DB_URL.split('://')[0]})")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
