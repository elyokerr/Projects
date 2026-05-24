"""End-to-end smoke test for the monthly_refresh flow against fixture data.

Gated by ``RUN_SLOW=1`` because the flow includes an Evidence ``npm run build``
step that can take 5+ minutes. To run:

    $env:RUN_SLOW="1"; .venv\\Scripts\\python.exe -m pytest tests/test_flows_smoke.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow end-to-end smoke; set RUN_SLOW=1 to run",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "housing.duckdb"


@pytest.fixture(scope="module")
def fresh_duckdb():
    """Remove the duckdb file before the test so we start clean. Module-scoped
    so the smoke runs exactly once."""
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    yield
    # Don't tear down — keep the populated DB for inspection / Evidence build.


@pytest.mark.slow
def test_monthly_refresh_fixture_mode_populates_warehouse(fresh_duckdb):
    from flows.monthly_refresh import monthly_refresh

    result = monthly_refresh(target="duckdb", mode="fixture")

    assert isinstance(result, dict)
    assert result["target"] == "duckdb"

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        ct = con.sql("SELECT COUNT(*) FROM main.fct_transactions").fetchone()[0]
        assert ct > 0, f"fct_transactions empty after refresh: {ct} rows"

        ct = con.sql(
            "SELECT COUNT(*) FROM main.mart_premium_to_benchmark"
        ).fetchone()[0]
        assert ct > 0, f"mart_premium_to_benchmark empty after refresh: {ct} rows"
    finally:
        con.close()
