"""Tests for the warehouse loader."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from flows.tasks.load_warehouse import BigQueryFreeTierExhausted, load_parquet_to_duckdb


def test_load_creates_table(tmp_path):
    pq = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(pq)
    db = tmp_path / "w.duckdb"
    load_parquet_to_duckdb(pq, db, schema="raw", table="ppd")
    con = duckdb.connect(str(db))
    assert con.sql("SELECT count(*) FROM raw.ppd").fetchone()[0] == 2


def test_load_is_idempotent_via_unique_key(tmp_path):
    pq = tmp_path / "x.parquet"
    pd.DataFrame({"id": ["a", "b"], "v": [1, 2]}).to_parquet(pq)
    db = tmp_path / "w.duckdb"
    load_parquet_to_duckdb(pq, db, schema="raw", table="ppd", unique_key="id")
    load_parquet_to_duckdb(pq, db, schema="raw", table="ppd", unique_key="id")
    con = duckdb.connect(str(db))
    assert con.sql("SELECT count(*) FROM raw.ppd").fetchone()[0] == 2


def test_load_rolls_back_on_error(tmp_path):
    bad = tmp_path / "x.parquet"
    bad.write_bytes(b"not a parquet")
    db = tmp_path / "w.duckdb"
    with pytest.raises(Exception):  # noqa: B017
        load_parquet_to_duckdb(bad, db, schema="raw", table="ppd")
    con = duckdb.connect(str(db))
    tables = con.sql("SHOW TABLES").fetchall()
    assert ("ppd",) not in tables


def test_load_to_bigquery_with_unique_key_does_merge_and_drops_staging(tmp_path):
    """Unit test: BigQuery append+unique_key path loads staging, runs MERGE, drops staging."""
    fake_parquet = tmp_path / "ppd.parquet"
    fake_parquet.write_bytes(b"placeholder")

    fake_client = MagicMock()
    fake_load_job = MagicMock()
    fake_query_job = MagicMock()
    fake_client.load_table_from_file.return_value = fake_load_job
    fake_client.query.return_value = fake_query_job

    fake_bigquery = MagicMock()
    fake_bigquery.Client.return_value = fake_client
    fake_bigquery.SourceFormat.PARQUET = "PARQUET"
    fake_bigquery.WriteDisposition.WRITE_TRUNCATE = "WRITE_TRUNCATE"
    fake_bigquery.WriteDisposition.WRITE_APPEND = "WRITE_APPEND"
    fake_bigquery.LoadJobConfig = MagicMock()

    fake_gcloud = MagicMock()
    fake_gcloud.bigquery = fake_bigquery
    fake_exceptions = MagicMock()
    fake_exceptions.Forbidden = type("Forbidden", (Exception,), {})

    with patch.dict(
        sys.modules,
        {
            "google": MagicMock(),
            "google.cloud": fake_gcloud,
            "google.cloud.bigquery": fake_bigquery,
            "google.api_core": MagicMock(exceptions=fake_exceptions),
            "google.api_core.exceptions": fake_exceptions,
        },
    ):
        from flows.tasks.load_warehouse import load_parquet_to_bigquery

        load_parquet_to_bigquery(
            fake_parquet,
            "proj",
            "ds",
            "ppd",
            unique_key="id",
            mode="append",
        )

    # Client constructed once
    assert fake_bigquery.Client.call_count == 1

    # Load was issued to the staging table
    load_call = fake_client.load_table_from_file.call_args
    assert load_call.args[1] == "proj.ds.raw_inc_ppd"

    # query() was called with MERGE referencing the target table, and DROP TABLE
    # referencing the staging table.
    queries = [c.args[0] for c in fake_client.query.call_args_list]
    assert any("MERGE" in q and "proj.ds.ppd" in q for q in queries)
    assert any("DROP TABLE" in q and "raw_inc_ppd" in q for q in queries)


def test_router_duckdb_target_calls_duckdb_loader(tmp_path):
    from flows.tasks import warehouse_router

    db_path = tmp_path / "w.duckdb"
    with patch.object(warehouse_router, "load_parquet_to_duckdb") as mock_duck, \
            patch.object(warehouse_router, "load_parquet_to_bigquery") as mock_bq:
        result = warehouse_router.load_to_warehouse.fn(
            tmp_path / "x.parquet",
            target_pref="duckdb",
            schema="raw",
            table="ppd",
            unique_key="id",
            duckdb_path=db_path,
        )

    assert mock_duck.call_count == 1
    assert mock_bq.call_count == 0
    assert result == ("duckdb", db_path)


def test_router_bigquery_falls_back_on_quota(tmp_path):
    from flows.tasks import warehouse_router

    db_path = tmp_path / "w.duckdb"

    def _raise_quota(*args, **kwargs):
        raise BigQueryFreeTierExhausted("quotaExceeded")

    with patch.object(
        warehouse_router, "load_parquet_to_bigquery", side_effect=_raise_quota
    ) as mock_bq, patch.object(
        warehouse_router, "load_parquet_to_duckdb"
    ) as mock_duck:
        result = warehouse_router.load_to_warehouse.fn(
            tmp_path / "x.parquet",
            target_pref="bigquery",
            schema="raw",
            table="ppd",
            unique_key="id",
            duckdb_path=db_path,
            bq_project="proj",
            bq_dataset="ds",
        )

    assert mock_bq.call_count == 1
    assert mock_duck.call_count == 1
    assert result == ("duckdb", db_path)
