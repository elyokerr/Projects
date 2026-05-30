"""Pytest configuration: make `from src.x import y` work regardless of cwd.

This file is auto-detected by pytest. With it in place, you can run the
suite from the project root with `pytest tests` or from inside any
subdirectory and the imports still resolve.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def sample_df():
    fixture = Path(__file__).resolve().parent / "fixtures" / "sample_transactions.parquet"
    return pd.read_parquet(fixture)
