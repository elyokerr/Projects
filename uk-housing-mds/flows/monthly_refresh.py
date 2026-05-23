"""Top-level monthly refresh flow.

Pipeline: ingest (PPD + NSPL + HPI) -> GE landing checkpoint -> warehouse load
-> dbt build -> Evidence build.

NSPL and HPI are loaded directly via ``load_parquet_to_duckdb`` because the
quota-aware ``warehouse_router.load_to_warehouse`` does not accept a ``mode``
kwarg. Trade-off: NSPL/HPI loads bypass the BigQuery-first router, but in the
free-tier setup ``target`` defaults to ``duckdb`` so this is moot. PPD still
goes through the router with ``unique_key`` for idempotent incremental loads.
"""

from __future__ import annotations

import os
from pathlib import Path

from prefect import flow, get_run_logger

from flows.tasks.build_evidence import build_evidence
from flows.tasks.ingest_hpi import ingest_hpi
from flows.tasks.ingest_nspl import ingest_nspl
from flows.tasks.ingest_ppd import ingest_ppd
from flows.tasks.load_warehouse import load_parquet_to_duckdb
from flows.tasks.run_data_quality import run_landing_checkpoint
from flows.tasks.run_dbt import run_dbt
from flows.tasks.warehouse_router import DEFAULT_DUCKDB_PATH, load_to_warehouse

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@flow(name="uk-housing-mds:monthly-refresh")
def monthly_refresh(target: str = "duckdb", mode: str = "increment") -> dict:
    """Orchestrate the monthly refresh end-to-end."""
    log = get_run_logger()
    log.info("monthly_refresh target=%s mode=%s", target, mode)

    # Flow mode "fixture" propagates to all three ingest tasks; otherwise each
    # task uses its own default (PPD "increment", NSPL "quarterly", HPI "monthly").
    if mode == "fixture":
        ppd_path = ingest_ppd(PROJECT_ROOT / "data" / "landing" / "ppd", mode="fixture")
        nspl_path = ingest_nspl(PROJECT_ROOT / "data" / "landing" / "nspl", mode="fixture")
        hpi_path = ingest_hpi(PROJECT_ROOT / "data" / "landing" / "hpi", mode="fixture")
    else:
        ppd_path = ingest_ppd(PROJECT_ROOT / "data" / "landing" / "ppd", mode=mode)
        nspl_path = ingest_nspl(PROJECT_ROOT / "data" / "landing" / "nspl")
        hpi_path = ingest_hpi(PROJECT_ROOT / "data" / "landing" / "hpi")

    run_landing_checkpoint("landing_all")

    effective_target, _ = load_to_warehouse(
        ppd_path,
        target_pref=target,
        schema="raw",
        table="ppd",
        unique_key="transaction_unique_id",
    )

    load_parquet_to_duckdb(
        nspl_path,
        db_path=DEFAULT_DUCKDB_PATH,
        schema="raw",
        table="nspl",
        mode="truncate",
    )
    load_parquet_to_duckdb(
        hpi_path,
        db_path=DEFAULT_DUCKDB_PATH,
        schema="raw",
        table="hpi",
        mode="truncate",
    )

    # dbt's profiles.yml resolves `../data/duckdb/...` from CWD, not from
    # --project-dir. Pin DUCKDB_PATH absolutely so the env_var template wins.
    os.environ.setdefault(
        "DUCKDB_PATH",
        str(PROJECT_ROOT / "data" / "duckdb" / "housing.duckdb"),
    )
    run_dbt("build", target=effective_target, project_dir=PROJECT_ROOT / "dbt_project")

    build_dir = build_evidence(
        evidence_dir=PROJECT_ROOT / "evidence",
        warehouse_path=PROJECT_ROOT / "data" / "duckdb" / "housing.duckdb",
    )

    return {
        "target": effective_target,
        "mode": mode,
        "ppd_path": str(ppd_path),
        "evidence_built": build_dir is not None,
    }


if __name__ == "__main__":
    monthly_refresh(target=os.environ.get("DBT_TARGET", "duckdb"))
