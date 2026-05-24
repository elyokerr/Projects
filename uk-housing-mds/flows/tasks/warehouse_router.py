"""Quota-aware warehouse-target router: BigQuery with DuckDB fallback."""

from __future__ import annotations

from pathlib import Path

import prefect

from flows.tasks.load_warehouse import (
    BigQueryFreeTierExhausted,
    load_parquet_to_bigquery,
    load_parquet_to_duckdb,
)

DEFAULT_DUCKDB_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "duckdb" / "housing.duckdb"
)


@prefect.task
def load_to_warehouse(
    parquet_path: Path,
    *,
    target_pref: str,
    schema: str = "raw",
    table: str,
    unique_key: str | None = None,
    mode: str = "append",
    duckdb_path: Path | None = None,
    bq_project: str | None = None,
    bq_dataset: str | None = None,
) -> tuple[str, Path | None]:
    """Route a parquet load to the preferred warehouse, falling back to DuckDB on BQ quota."""
    effective_duckdb_path = duckdb_path or DEFAULT_DUCKDB_PATH

    if target_pref == "bigquery":
        try:
            load_parquet_to_bigquery(
                parquet_path,
                project=bq_project,
                dataset=bq_dataset,
                table=table,
                unique_key=unique_key,
                mode=mode,
            )
            return ("bigquery", None)
        except BigQueryFreeTierExhausted as exc:
            try:
                log = prefect.get_run_logger()
                log.warning(
                    "BigQuery free tier exhausted (%s); falling back to DuckDB.", exc
                )
            except Exception:
                pass
            load_parquet_to_duckdb(
                parquet_path,
                db_path=effective_duckdb_path,
                schema=schema,
                table=table,
                unique_key=unique_key,
            )
            return ("duckdb", effective_duckdb_path)

    load_parquet_to_duckdb(
        parquet_path,
        db_path=effective_duckdb_path,
        schema=schema,
        table=table,
        unique_key=unique_key,
    )
    return ("duckdb", effective_duckdb_path)
