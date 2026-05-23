"""Idempotent Parquet → warehouse loader. DuckDB + BigQuery."""

from __future__ import annotations

from pathlib import Path

import duckdb
from prefect import task


class BigQueryFreeTierExhausted(RuntimeError):
    """Raised when a BigQuery load is rejected because the free tier quota is exhausted."""

    pass


def load_parquet_to_duckdb(
    parquet_path: Path,
    db_path: Path,
    *,
    schema: str,
    table: str,
    unique_key: str | None = None,
) -> None:
    parquet_path = Path(parquet_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    con.execute("BEGIN")
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        exists = con.sql(
            f"SELECT 1 FROM information_schema.tables "
            f"WHERE table_schema='{schema}' AND table_name='{table}'"
        ).fetchone()
        if not exists:
            con.execute(
                f"CREATE TABLE {schema}.{table} AS "
                f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
            )
        else:
            if unique_key:
                con.execute(
                    f"CREATE TEMP TABLE _incoming AS "
                    f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
                )
                con.execute(
                    f"DELETE FROM {schema}.{table} t "
                    f"WHERE t.{unique_key} IN (SELECT {unique_key} FROM _incoming)"
                )
                con.execute(
                    f"INSERT INTO {schema}.{table} SELECT * FROM _incoming"
                )
                con.execute("DROP TABLE _incoming")
            else:
                con.execute(
                    f"INSERT INTO {schema}.{table} "
                    f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
                )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


load_parquet_to_duckdb_task = task(retries=2, retry_delay_seconds=30)(
    load_parquet_to_duckdb
)


def load_parquet_to_bigquery(
    parquet_path: Path,
    project: str,
    dataset: str,
    table: str,
    *,
    unique_key: str | None = None,
    mode: str = "append",
) -> None:
    """Load a Parquet file into BigQuery, optionally with MERGE-based dedupe.

    Modes:
        - "truncate": full-refresh (WRITE_TRUNCATE) into `{project}.{dataset}.{table}`
        - "append" with no unique_key: WRITE_APPEND into target
        - "append" with unique_key: load to staging `raw_inc_{table}` then MERGE into target
    """
    from google.api_core import exceptions as gcp_exceptions
    from google.cloud import bigquery

    parquet_path = Path(parquet_path)
    target_fqn = f"{project}.{dataset}.{table}"

    try:
        client = bigquery.Client(project=project)

        if mode == "truncate":
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
            with parquet_path.open("rb") as f:
                job = client.load_table_from_file(
                    f, target_fqn, job_config=job_config
                )
            job.result()
            return

        if mode == "append" and not unique_key:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            with parquet_path.open("rb") as f:
                job = client.load_table_from_file(
                    f, target_fqn, job_config=job_config
                )
            job.result()
            return

        # append + unique_key: load to staging, MERGE, drop staging
        staging_table = f"raw_inc_{table}"
        staging_fqn = f"{project}.{dataset}.{staging_table}"
        staging_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with parquet_path.open("rb") as f:
            load_job = client.load_table_from_file(
                f, staging_fqn, job_config=staging_config
            )
        load_job.result()

        merge_sql = (
            f"MERGE `{target_fqn}` T "
            f"USING `{staging_fqn}` S "
            f"ON T.{unique_key} = S.{unique_key} "
            f"WHEN MATCHED THEN UPDATE SET "
            f"  {unique_key} = S.{unique_key} "
            f"WHEN NOT MATCHED THEN INSERT ROW"
        )
        merge_job = client.query(merge_sql)
        merge_job.result()

        drop_sql = f"DROP TABLE `{staging_fqn}`"
        drop_job = client.query(drop_sql)
        drop_job.result()
    except gcp_exceptions.Forbidden as e:
        raise BigQueryFreeTierExhausted(str(e)) from e
    except Exception as e:
        if "quotaExceeded" in str(e):
            raise BigQueryFreeTierExhausted(str(e)) from e
        raise
