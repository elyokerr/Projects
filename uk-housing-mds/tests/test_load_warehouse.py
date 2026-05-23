"""Tests for the warehouse loader."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from flows.tasks.load_warehouse import load_parquet_to_duckdb


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
