"""
Centralised configuration for the SaaS Churn Prediction project.
All paths, database settings, and model parameters are managed here.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"

# Raw data file
RAW_CSV_PATH = RAW_DATA_DIR / "telco_churn_raw.csv"

# ─── Database Configuration ──────────────────────────────────────
# Uses environment variables with sensible defaults for local dev
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "saas_churn"),
    "user": os.getenv("DB_USER", "churn_user"),
    "password": os.getenv("DB_PASSWORD", "churn_pass"),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# SQLite fallback for lightweight/local development without Docker
SQLITE_URL = f"sqlite:///{DATA_DIR / 'churn.db'}"

# Toggle: set USE_POSTGRES=true in env for PostgreSQL, otherwise SQLite
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"
ACTIVE_DB_URL = DATABASE_URL if USE_POSTGRES else SQLITE_URL

# ─── Model Configuration ────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "churn"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "saas-churn-prediction"
