"""
Centralized configuration for the SaaS Churn Prediction project.

Every path, database setting, and model hyperparameter that more than one
module needs to know about lives here. Reading a single file is much easier
than hunting through the codebase for hard-coded constants.

The module also handles the SQLite-vs-PostgreSQL switch through the
USE_POSTGRES environment variable, which makes the same code run locally
on a laptop and inside Docker Compose without modification.
"""

import os
from pathlib import Path


# --- Project paths --------------------------------------------------------

# This file lives at src/config.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "src" / "sql"

RAW_CSV_PATH = RAW_DATA_DIR / "telco_churn_raw.csv"


# --- Database configuration -----------------------------------------------

# PostgreSQL connection settings. The defaults match the values declared in
# docker-compose.yml, so containers wire up without any extra configuration.
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

# SQLite fallback for local development. The same SQL features.sql file
# works against both engines because it sticks to standard CTEs and avoids
# vendor-specific syntax.
SQLITE_URL = f"sqlite:///{DATA_DIR / 'churn.db'}"

# Set USE_POSTGRES=true in the environment to switch backends.
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"
ACTIVE_DB_URL = DATABASE_URL if USE_POSTGRES else SQLITE_URL


# --- Model configuration --------------------------------------------------

RANDOM_STATE = 42        # Fixed for reproducible train/test splits and CV
TEST_SIZE = 0.2          # 80/20 stratified split
TARGET_COLUMN = "churn"

# MLflow stores experiment metadata under mlruns/ by default. Override
# MLFLOW_TRACKING_URI in the environment to point at a remote server.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file://{PROJECT_ROOT / 'mlruns'}",
)
MLFLOW_EXPERIMENT_NAME = "saas-churn-prediction"
