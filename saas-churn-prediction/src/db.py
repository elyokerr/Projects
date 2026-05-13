"""
Database connection helpers.

A thin wrapper around SQLAlchemy that produces an engine pointing at either
SQLite (local development) or PostgreSQL (containerized stack) based on
the USE_POSTGRES environment variable. The rest of the codebase calls
get_engine() and never has to care which backend is active.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import ACTIVE_DB_URL, USE_POSTGRES


def get_engine():
    """Return a SQLAlchemy engine configured for the active backend.

    SQLite needs check_same_thread=False so the engine can be shared
    across threads in environments like Streamlit or pytest. PostgreSQL
    handles that natively and needs no extra connect_args.
    """
    connect_args = {} if USE_POSTGRES else {"check_same_thread": False}
    return create_engine(ACTIVE_DB_URL, echo=False, connect_args=connect_args)


def get_session():
    """Create a new database session bound to a fresh engine."""
    Session = sessionmaker(bind=get_engine())
    return Session()


def test_connection() -> bool:
    """Verify connectivity to the active database.

    Returns True on success, False on failure. Useful as a smoke test
    when bringing up a new environment.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        backend = ACTIVE_DB_URL.split("://")[0]
        print(f"Database connection successful ({backend})")
        return True
    except Exception as exc:
        print(f"Database connection failed: {exc}")
        return False


if __name__ == "__main__":
    test_connection()
