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

# Load .env so the gated e2e eval picks up GROQ_API_KEY / GOOGLE_API_KEY
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def esco_matcher():
    """Module-scoped EscoMatcher so the fastembed model loads once per module."""
    from src.data.esco_loader import EscoIndex
    from src.skills.esco_matcher import EscoMatcher

    return EscoMatcher(EscoIndex.load())
