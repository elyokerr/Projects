"""Load landing fixture parquets into the local DuckDB warehouse under schema `raw`.

Idempotent: PPD uses unique_key dedupe; NSPL and HPI are truncated and reloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flows.tasks.load_warehouse import load_parquet_to_duckdb  # noqa: E402

LANDING = ROOT / "data" / "landing"
DB_PATH = ROOT / "data" / "duckdb" / "housing.duckdb"


def _truncate(schema: str, table: str) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
    finally:
        con.close()


def main() -> None:
    ppd_pq = LANDING / "ppd" / "fixture.parquet"
    nspl_pq = LANDING / "nspl" / "fixture.parquet"
    hpi_pq = LANDING / "hpi" / "fixture.parquet"

    load_parquet_to_duckdb(
        ppd_pq, DB_PATH, schema="raw", table="ppd",
        unique_key="transaction_unique_id",
    )

    _truncate("raw", "nspl")
    load_parquet_to_duckdb(nspl_pq, DB_PATH, schema="raw", table="nspl")

    _truncate("raw", "hpi")
    load_parquet_to_duckdb(hpi_pq, DB_PATH, schema="raw", table="hpi")

    print(f"Loaded fixtures into {DB_PATH}", flush=True)


if __name__ == "__main__":
    main()
